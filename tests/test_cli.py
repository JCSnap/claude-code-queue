"""
Tests for the CLI module — Pass 7.

Covers: argument parsing, global option forwarding, command dispatch,
output format (text and JSON), bank sub-commands, prompt-box passthrough,
and exit codes for every command.

Storage commands mock ``QueueStorage``; commands that invoke claude mock ``QueueManager``.
Commands are exercised by patching ``sys.argv`` and calling ``main()``.
"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pytest

from claude_code_queue.cli import main
from claude_code_queue.models import (
    PromptStatus,
    QueuedPrompt,
    QueueState,
    RateLimitInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    prompts=None,
    last_processed=None,
    total_processed=0,
    failed_count=0,
    rate_limited_count=0,
    current_rate_limit=None,
) -> QueueState:
    """Build a QueueState with sensible defaults."""
    return QueueState(
        prompts=prompts if prompts is not None else [],
        last_processed=last_processed,
        total_processed=total_processed,
        failed_count=failed_count,
        rate_limited_count=rate_limited_count,
        current_rate_limit=current_rate_limit,
    )


def _make_template(
    name="my-tmpl",
    title="My Template",
    priority=0,
    working_directory=".",
    estimated_tokens=None,
    modified=None,
):
    """Build a bank-template dict like QueueStorage returns."""
    if modified is None:
        modified = datetime(2026, 3, 1, 10, 0, 0)
    return {
        "name": name,
        "title": title,
        "priority": priority,
        "working_directory": working_directory,
        "estimated_tokens": estimated_tokens,
        "modified": modified,
    }


# ===========================================================================
# No Command
# ===========================================================================

class TestNoCommand:
    def test_no_command_returns_1(self):
        with patch("sys.argv", ["claude-queue"]):
            code = main()
        assert code == 1


# ===========================================================================
# Global Options — forwarded to QueueManager()
# ===========================================================================

class TestGlobalOptions:
    """Each test invokes a real subcommand (test) to force QueueManager init."""

    def _run(self, *argv):
        """Run main() and return (exit_code, QueueManager class mock)."""
        with patch("sys.argv", ["claude-queue"] + list(argv)):
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.claude_interface.test_connection.return_value = (True, "ok")
                mock_cls.return_value = mgr
                code = main()
                return code, mock_cls

    def test_default_storage_dir(self):
        _, mock_cls = self._run("test")
        _, kwargs = mock_cls.call_args
        assert kwargs["storage_dir"] == "~/.claude-queue"

    def test_custom_storage_dir(self):
        _, mock_cls = self._run("--storage-dir", "/tmp/q", "test")
        _, kwargs = mock_cls.call_args
        assert kwargs["storage_dir"] == "/tmp/q"

    def test_default_claude_command(self):
        _, mock_cls = self._run("test")
        _, kwargs = mock_cls.call_args
        assert kwargs["claude_command"] == "claude"

    def test_custom_claude_command(self):
        _, mock_cls = self._run("--claude-command", "my-claude", "test")
        _, kwargs = mock_cls.call_args
        assert kwargs["claude_command"] == "my-claude"

    def test_default_check_interval(self):
        _, mock_cls = self._run("test")
        _, kwargs = mock_cls.call_args
        assert kwargs["check_interval"] == 30

    def test_custom_check_interval(self):
        _, mock_cls = self._run("--check-interval", "60", "test")
        _, kwargs = mock_cls.call_args
        assert kwargs["check_interval"] == 60

    def test_default_timeout(self):
        _, mock_cls = self._run("test")
        _, kwargs = mock_cls.call_args
        assert kwargs["timeout"] == 3600

    def test_custom_timeout(self):
        _, mock_cls = self._run("--timeout", "7200", "test")
        _, kwargs = mock_cls.call_args
        assert kwargs["timeout"] == 7200


# ===========================================================================
# `add` Command
# ===========================================================================

class TestAddCommand:
    def _run_add(self, *extra_args, success=True):
        with patch("sys.argv", ["claude-queue", "add", "hello world"] + list(extra_args)):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage._save_single_prompt.return_value = success
                mock_cls.return_value = storage
                code = main()
                return code, storage

    def test_add_prompt_content(self):
        _, storage = self._run_add()
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.content == "hello world"

    def test_add_default_priority_zero(self):
        _, storage = self._run_add()
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.priority == 0

    def test_add_long_priority_flag(self):
        _, storage = self._run_add("--priority", "5")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.priority == 5

    def test_add_short_priority_flag(self):
        _, storage = self._run_add("-p", "3")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.priority == 3

    def test_add_default_working_dir_dot(self):
        _, storage = self._run_add()
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.working_directory == "."

    def test_add_long_working_dir_flag(self):
        _, storage = self._run_add("--working-dir", "/some/path")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.working_directory == "/some/path"

    def test_add_short_working_dir_flag(self):
        _, storage = self._run_add("-d", "/some/path")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.working_directory == "/some/path"

    def test_add_context_files_long_flag(self):
        _, storage = self._run_add("--context-files", "a.py", "b.py")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.context_files == ["a.py", "b.py"]

    def test_add_context_files_short_flag(self):
        _, storage = self._run_add("-f", "a.py")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.context_files == ["a.py"]

    def test_add_default_context_files_empty(self):
        _, storage = self._run_add()
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.context_files == []

    def test_add_max_retries_long_flag(self):
        _, storage = self._run_add("--max-retries", "5")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.max_retries == 5

    def test_add_max_retries_short_flag(self):
        _, storage = self._run_add("-r", "2")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.max_retries == 2

    def test_add_default_max_retries_three(self):
        _, storage = self._run_add()
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.max_retries == 3

    def test_add_estimated_tokens_long_flag(self):
        _, storage = self._run_add("--estimated-tokens", "1500")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.estimated_tokens == 1500

    def test_add_estimated_tokens_short_flag(self):
        _, storage = self._run_add("-t", "1000")
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.estimated_tokens == 1000

    def test_add_default_estimated_tokens_none(self):
        _, storage = self._run_add()
        prompt = storage._save_single_prompt.call_args[0][0]
        assert prompt.estimated_tokens is None

    def test_add_returns_zero_on_success(self):
        code, _ = self._run_add(success=True)
        assert code == 0

    def test_add_returns_one_on_failure(self):
        code, _ = self._run_add(success=False)
        assert code == 1

    def test_add_passes_storage_dir_to_storage(self):
        with patch("sys.argv", ["claude-queue", "--storage-dir", "/custom/path", "add", "hello"]):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                mock_cls.return_value._save_single_prompt.return_value = True
                main()
                mock_cls.assert_called_once()  # guard: fails clearly if storage was never constructed
                _, kwargs = mock_cls.call_args
                assert kwargs["storage_dir"] == "/custom/path"


# ===========================================================================
# `template` Command
# ===========================================================================

class TestTemplateCommand:
    def _run_template(self, *extra_args, path="/some/path/my-template.md"):
        with patch("sys.argv", ["claude-queue", "template", "my-template"] + list(extra_args)):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.create_prompt_template.return_value = path
                mock_cls.return_value = storage
                code = main()
                return code, storage

    def test_template_calls_create_with_filename(self):
        _, storage = self._run_template()
        storage.create_prompt_template.assert_called_once_with("my-template", 0)

    def test_template_default_priority_zero(self):
        _, storage = self._run_template()
        _, priority = storage.create_prompt_template.call_args[0]
        assert priority == 0

    def test_template_long_priority_flag(self):
        _, storage = self._run_template("--priority", "2")
        _, priority = storage.create_prompt_template.call_args[0]
        assert priority == 2

    def test_template_short_priority_flag(self):
        _, storage = self._run_template("-p", "1")
        _, priority = storage.create_prompt_template.call_args[0]
        assert priority == 1

    def test_template_prints_created_path(self, capsys):
        self._run_template(path="/some/path/my-template.md")
        captured = capsys.readouterr()
        assert "/some/path/my-template.md" in captured.out

    def test_template_prints_edit_hint(self, capsys):
        self._run_template()
        captured = capsys.readouterr()
        assert "Edit the file" in captured.out

    def test_template_returns_zero(self):
        code, _ = self._run_template()
        assert code == 0


# ===========================================================================
# `status` Command
# ===========================================================================

class TestStatusCommand:
    def _run_status(self, *extra_args, state=None):
        if state is None:
            state = _make_state()
        with patch("sys.argv", ["claude-queue", "status"] + list(extra_args)):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.load_queue_state.return_value = state
                mock_cls.return_value = storage
                code = main()
                return code

    def test_status_default_text_output(self, capsys):
        self._run_status()
        captured = capsys.readouterr()
        assert "Claude Code Queue Status" in captured.out

    def test_status_shows_total_prompts(self, capsys):
        state = _make_state(prompts=[QueuedPrompt(content="x")])
        self._run_status(state=state)
        captured = capsys.readouterr()
        assert "Total prompts: 1" in captured.out

    def test_status_shows_total_processed(self, capsys):
        state = _make_state(total_processed=7)
        self._run_status(state=state)
        captured = capsys.readouterr()
        assert "Total processed: 7" in captured.out

    def test_status_shows_failed_count(self, capsys):
        state = _make_state(failed_count=3)
        self._run_status(state=state)
        captured = capsys.readouterr()
        assert "Failed count: 3" in captured.out

    def test_status_json_flag_outputs_valid_json(self, capsys):
        self._run_status("--json")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)  # must not raise
        assert isinstance(parsed, dict)

    def test_status_json_has_total_prompts(self, capsys):
        self._run_status("--json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total_prompts" in data

    def test_status_json_has_status_counts(self, capsys):
        self._run_status("--json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "status_counts" in data

    def test_status_detailed_long_flag_shows_prompts(self, capsys):
        p = QueuedPrompt(content="fix the bug", id="abc12345")
        state = _make_state(prompts=[p])
        self._run_status("--detailed", state=state)
        captured = capsys.readouterr()
        assert "abc12345" in captured.out

    def test_status_detailed_short_flag(self, capsys):
        p = QueuedPrompt(content="fix the bug", id="xyz99999")
        state = _make_state(prompts=[p])
        self._run_status("-d", state=state)
        captured = capsys.readouterr()
        assert "xyz99999" in captured.out

    def test_status_shows_rate_limit_reset_time_when_rate_limited(self, capsys):
        reset_dt = datetime(2026, 3, 1, 15, 30, 0)
        rl = RateLimitInfo(is_rate_limited=True, reset_time=reset_dt)
        state = _make_state(current_rate_limit=rl)
        self._run_status(state=state)
        captured = capsys.readouterr()
        assert "Rate limited until:" in captured.out

    def test_status_no_rate_limit_no_extra_line(self, capsys):
        self._run_status()
        captured = capsys.readouterr()
        assert "Rate limited until:" not in captured.out

    def test_status_last_processed_shown_when_set(self, capsys):
        last = datetime(2026, 3, 1, 12, 0, 0)
        state = _make_state(last_processed=last)
        self._run_status(state=state)
        captured = capsys.readouterr()
        assert "2026-03-01" in captured.out

    def test_status_no_crash_when_last_processed_none(self):
        state = _make_state(last_processed=None)
        code = self._run_status(state=state)
        assert code == 0

    def test_status_returns_zero(self):
        code = self._run_status()
        assert code == 0


# ===========================================================================
# `cancel` Command
# ===========================================================================

class TestCancelCommand:
    def _run_cancel(self, prompt_id, success=True):
        # Build a real QueueState containing the target prompt so that
        # state.get_prompt(prompt_id) returns it instead of None.
        target_prompt = QueuedPrompt(id=prompt_id, content="some content")
        state = _make_state(prompts=[target_prompt])

        with patch("sys.argv", ["claude-queue", "cancel", prompt_id]):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.load_queue_state.return_value = state
                # Note: call_args captures a reference to the QueuedPrompt dataclass,
                # not a snapshot. Assertions on call_args[0][0] see the object's state
                # at assertion time. cmd_cancel does not modify the prompt after calling
                # _save_single_prompt, so this is safe.
                storage._save_single_prompt.return_value = success
                mock_cls.return_value = storage
                code = main()
                return code, storage

    def test_cancel_marks_prompt_cancelled_and_saves(self):
        _, storage = self._run_cancel("abc123")
        storage._save_single_prompt.assert_called_once()
        saved_prompt = storage._save_single_prompt.call_args[0][0]
        assert saved_prompt.id == "abc123"
        assert saved_prompt.status == PromptStatus.CANCELLED

    def test_cancel_returns_zero_on_success(self):
        code, _ = self._run_cancel("abc123", success=True)
        assert code == 0

    def test_cancel_returns_one_on_failure(self):
        code, _ = self._run_cancel("abc123", success=False)
        assert code == 1

    def test_cancel_prompt_not_found_returns_one(self):
        state = _make_state(prompts=[])  # target ID absent from queue
        with patch("sys.argv", ["claude-queue", "cancel", "abc123"]):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.load_queue_state.return_value = state
                mock_cls.return_value = storage
                code = main()
        assert code == 1

    def test_cancel_executing_prompt_returns_one(self):
        target_prompt = QueuedPrompt(
            id="abc123", content="x", status=PromptStatus.EXECUTING
        )
        state = _make_state(prompts=[target_prompt])
        with patch("sys.argv", ["claude-queue", "cancel", "abc123"]):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.load_queue_state.return_value = state
                mock_cls.return_value = storage
                code = main()
        assert code == 1


# ===========================================================================
# `list` Command
# ===========================================================================

class TestListCommand:
    _ALL_PROMPTS = [
        QueuedPrompt(id="p1", content="queued prompt", status=PromptStatus.QUEUED),
        QueuedPrompt(id="p2", content="exec prompt", status=PromptStatus.EXECUTING),
        QueuedPrompt(id="p3", content="done prompt", status=PromptStatus.COMPLETED),
        QueuedPrompt(id="p4", content="fail prompt", status=PromptStatus.FAILED),
        QueuedPrompt(id="p5", content="cancel prompt", status=PromptStatus.CANCELLED),
        QueuedPrompt(id="p6", content="rl prompt", status=PromptStatus.RATE_LIMITED),
    ]

    def _run_list(self, *extra_args, prompts=None):
        p = prompts if prompts is not None else self._ALL_PROMPTS
        state = _make_state(prompts=p)
        with patch("sys.argv", ["claude-queue", "list"] + list(extra_args)):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.load_queue_state.return_value = state
                mock_cls.return_value = storage
                code = main()
                return code

    def test_list_default_shows_all_prompts(self, capsys):
        self._run_list()
        out = capsys.readouterr().out
        assert "p1" in out
        assert "p6" in out

    def test_list_status_filter_queued(self, capsys):
        self._run_list("--status", "queued")
        out = capsys.readouterr().out
        assert "p1" in out
        assert "p2" not in out

    def test_list_status_filter_executing(self, capsys):
        self._run_list("--status", "executing")
        out = capsys.readouterr().out
        assert "p2" in out
        assert "p1" not in out

    def test_list_status_filter_completed(self, capsys):
        self._run_list("--status", "completed")
        out = capsys.readouterr().out
        assert "p3" in out
        assert "p1" not in out

    def test_list_status_filter_failed(self, capsys):
        self._run_list("--status", "failed")
        out = capsys.readouterr().out
        assert "p4" in out
        assert "p1" not in out

    def test_list_status_filter_cancelled(self, capsys):
        self._run_list("--status", "cancelled")
        out = capsys.readouterr().out
        assert "p5" in out
        assert "p1" not in out

    def test_list_status_filter_rate_limited(self, capsys):
        self._run_list("--status", "rate_limited")
        out = capsys.readouterr().out
        assert "p6" in out
        assert "p1" not in out

    def test_list_invalid_status_is_argparse_error(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["claude-queue", "list", "--status", "invalid_value"]):
                with patch("claude_code_queue.cli.QueueStorage"):
                    main()
        assert exc_info.value.code != 0

    def test_list_json_flag_outputs_valid_json(self, capsys):
        self._run_list("--json")
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)

    def test_list_json_item_has_id(self, capsys):
        self._run_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("id" in item for item in data)

    def test_list_json_item_has_content(self, capsys):
        self._run_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("content" in item for item in data)

    def test_list_json_item_has_status(self, capsys):
        self._run_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("status" in item for item in data)

    def test_list_json_item_has_priority(self, capsys):
        self._run_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("priority" in item for item in data)

    def test_list_json_item_has_working_directory(self, capsys):
        self._run_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("working_directory" in item for item in data)

    def test_list_json_item_has_created_at(self, capsys):
        self._run_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("created_at" in item for item in data)

    def test_list_json_item_has_retry_fields(self, capsys):
        self._run_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("retry_count" in item and "max_retries" in item for item in data)

    def test_list_empty_queue_prints_no_prompts_message(self, capsys):
        self._run_list(prompts=[])
        out = capsys.readouterr().out
        assert "No prompts found" in out

    def test_list_returns_zero(self):
        code = self._run_list()
        assert code == 0


# ===========================================================================
# `test` Command
# ===========================================================================

class TestTestCommand:
    def _run_test_cmd(self, is_working, message):
        with patch("sys.argv", ["claude-queue", "test"]):
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.claude_interface.test_connection.return_value = (is_working, message)
                mock_cls.return_value = mgr
                return main()

    def test_test_prints_message_from_connection(self, capsys):
        self._run_test_cmd(True, "Connection is working fine!")
        assert "Connection is working fine!" in capsys.readouterr().out

    def test_test_returns_zero_when_working(self):
        assert self._run_test_cmd(True, "OK") == 0

    def test_test_returns_one_when_failing(self):
        assert self._run_test_cmd(False, "Error: connection failed") == 1


# ===========================================================================
# `bank save` Command
# ===========================================================================

class TestBankSaveCommand:
    _BANK_PATH = "/bank/dir/my-template.md"

    def _run_bank_save(self, *extra_args):
        with patch("sys.argv", ["claude-queue", "bank", "save", "my-template"] + list(extra_args)):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.save_prompt_to_bank.return_value = self._BANK_PATH
                mock_cls.return_value = storage
                code = main()
                return code, storage

    def test_bank_save_calls_save_prompt_to_bank(self):
        _, storage = self._run_bank_save()
        storage.save_prompt_to_bank.assert_called_once_with("my-template", 0)

    def test_bank_save_default_priority_zero(self):
        _, storage = self._run_bank_save()
        _, priority = storage.save_prompt_to_bank.call_args[0]
        assert priority == 0

    def test_bank_save_long_priority_flag(self):
        _, storage = self._run_bank_save("--priority", "3")
        _, priority = storage.save_prompt_to_bank.call_args[0]
        assert priority == 3

    def test_bank_save_short_priority_flag(self):
        _, storage = self._run_bank_save("-p", "2")
        _, priority = storage.save_prompt_to_bank.call_args[0]
        assert priority == 2

    def test_bank_save_prints_path(self, capsys):
        self._run_bank_save()
        assert self._BANK_PATH in capsys.readouterr().out

    def test_bank_save_prints_edit_hint(self, capsys):
        self._run_bank_save()
        assert "Edit" in capsys.readouterr().out

    def test_bank_save_returns_zero(self):
        code, _ = self._run_bank_save()
        assert code == 0


# ===========================================================================
# `bank list` Command
# ===========================================================================

class TestBankListCommand:
    def _run_bank_list(self, *extra_args, templates=None):
        if templates is None:
            templates = [_make_template()]
        with patch("sys.argv", ["claude-queue", "bank", "list"] + list(extra_args)):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.list_bank_templates.return_value = templates
                mock_cls.return_value = storage
                code = main()
                return code

    def test_bank_list_empty_prints_no_templates_message(self, capsys):
        self._run_bank_list(templates=[])
        assert "No templates found in bank" in capsys.readouterr().out

    def test_bank_list_shows_template_names(self, capsys):
        self._run_bank_list(templates=[_make_template(name="cool-tmpl")])
        assert "cool-tmpl" in capsys.readouterr().out

    def test_bank_list_shows_title(self, capsys):
        self._run_bank_list(templates=[_make_template(title="Cool Title")])
        assert "Cool Title" in capsys.readouterr().out

    def test_bank_list_shows_priority(self, capsys):
        self._run_bank_list(templates=[_make_template(priority=5)])
        assert "5" in capsys.readouterr().out

    def test_bank_list_shows_working_directory(self, capsys):
        self._run_bank_list(templates=[_make_template(working_directory="/my/dir")])
        assert "/my/dir" in capsys.readouterr().out

    def test_bank_list_shows_estimated_tokens_when_set(self, capsys):
        self._run_bank_list(templates=[_make_template(estimated_tokens=1234)])
        assert "1234" in capsys.readouterr().out

    def test_bank_list_omits_estimated_tokens_when_none(self, capsys):
        self._run_bank_list(templates=[_make_template(estimated_tokens=None)])
        assert "Estimated tokens" not in capsys.readouterr().out

    def test_bank_list_shows_modified_timestamp(self, capsys):
        mod = datetime(2026, 3, 1, 10, 30, 0)
        self._run_bank_list(templates=[_make_template(modified=mod)])
        assert "2026-03-01" in capsys.readouterr().out

    def test_bank_list_json_flag_outputs_valid_json(self, capsys):
        self._run_bank_list("--json")
        parsed = json.loads(capsys.readouterr().out)  # must not raise
        assert parsed is not None

    def test_bank_list_json_is_array(self, capsys):
        self._run_bank_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)

    def test_bank_list_json_item_has_name(self, capsys):
        self._run_bank_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("name" in item for item in data)

    def test_bank_list_json_item_has_title(self, capsys):
        self._run_bank_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("title" in item for item in data)

    def test_bank_list_json_item_has_priority(self, capsys):
        self._run_bank_list("--json")
        data = json.loads(capsys.readouterr().out)
        assert all("priority" in item for item in data)

    def test_bank_list_returns_zero(self):
        code = self._run_bank_list()
        assert code == 0


# ===========================================================================
# `bank use` Command
# ===========================================================================

class TestBankUseCommand:
    def _run_bank_use(self, success=True):
        with patch("sys.argv", ["claude-queue", "bank", "use", "my-template"]):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                # use_bank_template must return a real QueuedPrompt so the CLI
                # doesn't short-circuit with "Template not found".
                # For success=False we still return a prompt but
                # _save_single_prompt returns False.
                storage.use_bank_template.return_value = QueuedPrompt(content="tmpl")
                storage._save_single_prompt.return_value = success
                mock_cls.return_value = storage
                return main(), storage

    def test_bank_use_calls_use_bank_template(self):
        _, storage = self._run_bank_use()
        storage.use_bank_template.assert_called_once_with("my-template")

    def test_bank_use_returns_zero_on_success(self):
        code, _ = self._run_bank_use(success=True)
        assert code == 0

    def test_bank_use_returns_one_when_save_fails(self):
        code, _ = self._run_bank_use(success=False)
        assert code == 1

    def test_bank_use_returns_one_when_template_not_found(self):
        with patch("sys.argv", ["claude-queue", "bank", "use", "my-template"]):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.use_bank_template.return_value = None  # template absent
                mock_cls.return_value = storage
                code = main()
        assert code == 1


# ===========================================================================
# `bank delete` Command
# ===========================================================================

class TestBankDeleteCommand:
    def _run_bank_delete(self, success=True):
        with patch("sys.argv", ["claude-queue", "bank", "delete", "my-template"]):
            with patch("claude_code_queue.cli.QueueStorage") as mock_cls:
                storage = MagicMock()
                storage.delete_bank_template.return_value = success
                mock_cls.return_value = storage
                return main(), storage

    def test_bank_delete_calls_delete_bank_template(self):
        _, storage = self._run_bank_delete()
        storage.delete_bank_template.assert_called_once_with("my-template")

    def test_bank_delete_returns_zero_on_success(self):
        code, _ = self._run_bank_delete(success=True)
        assert code == 0

    def test_bank_delete_returns_one_on_failure(self):
        code, _ = self._run_bank_delete(success=False)
        assert code == 1


# ===========================================================================
# `bank` Without Subcommand
# ===========================================================================

class TestBankNoSubcommand:
    """
    cmd_bank returns 1 before constructing any storage when no subcommand is given.
    The QueueManager patch is a no-op guard against accidental instantiation.
    """

    def _run_bank_no_subcommand(self):
        with patch("sys.argv", ["claude-queue", "bank"]):
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mock_cls.return_value = mgr
                return main()

    def test_bank_no_subcommand_returns_one(self):
        assert self._run_bank_no_subcommand() == 1

    def test_bank_no_subcommand_prints_error(self, capsys):
        self._run_bank_no_subcommand()
        assert "No bank operation specified" in capsys.readouterr().out

    def test_bank_no_subcommand_lists_available_operations(self, capsys):
        self._run_bank_no_subcommand()
        out = capsys.readouterr().out
        for op in ("save", "list", "use", "delete"):
            assert op in out


# ===========================================================================
# `prompt-box` Command
# ===========================================================================

class TestPromptBoxCommand:
    """
    cmd_prompt_box does not use QueueManager or QueueStorage. The patch is
    retained as a no-op guard against accidental QueueManager instantiation.
    shutil is imported inside the function, so we patch shutil.which directly.
    """

    def _run_prompt_box(
        self,
        *extra_args,
        which_returns="/usr/bin/prompt-box",
        run_returncode=0,
        raise_file_not_found=False,
    ):
        mock_result = MagicMock()
        mock_result.returncode = run_returncode

        if raise_file_not_found:
            mock_run = MagicMock(side_effect=FileNotFoundError())
        else:
            mock_run = MagicMock(return_value=mock_result)

        with patch("sys.argv", ["claude-queue", "prompt-box"] + list(extra_args)):
            with patch("claude_code_queue.cli.QueueManager"):
                with patch("shutil.which", return_value=which_returns) as mock_which:
                    # The CLI also calls os.path.exists(binary_path) after shutil.which;
                    # mock it to True so tests that expect success aren't blocked by the
                    # path being absent on the test machine.
                    with patch("claude_code_queue.cli.os.path.exists", return_value=True):
                        with patch("claude_code_queue.cli.subprocess.run", mock_run):
                            code = main()
                            return code, mock_which, mock_run

    def test_prompt_box_uses_shutil_which(self):
        _, mock_which, _ = self._run_prompt_box()
        mock_which.assert_called_once_with("prompt-box")

    def test_prompt_box_not_found_returns_one(self):
        with patch("sys.argv", ["claude-queue", "prompt-box"]):
            with patch("claude_code_queue.cli.QueueManager"):
                with patch("shutil.which", return_value=None):
                    with patch("claude_code_queue.cli.os.path.exists", return_value=False):
                        code = main()
        assert code == 1

    def test_prompt_box_not_found_prints_error(self, capsys):
        with patch("sys.argv", ["claude-queue", "prompt-box"]):
            with patch("claude_code_queue.cli.QueueManager"):
                with patch("shutil.which", return_value=None):
                    with patch("claude_code_queue.cli.os.path.exists", return_value=False):
                        main()
        out = capsys.readouterr().out
        assert "prompt-box binary not found" in out
        assert "rustup.rs" in out
        assert "force-reinstall" in out

    def test_prompt_box_executes_found_binary(self):
        _, _, mock_run = self._run_prompt_box()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/prompt-box"

    def test_prompt_box_passes_extra_args(self):
        _, _, mock_run = self._run_prompt_box("file.txt", "arg2")
        cmd = mock_run.call_args[0][0]
        assert "file.txt" in cmd
        assert "arg2" in cmd

    def test_prompt_box_returns_binary_exit_code(self):
        code, _, _ = self._run_prompt_box(run_returncode=42)
        assert code == 42

    def test_prompt_box_file_not_found_exception_returns_one(self):
        code, _, _ = self._run_prompt_box(raise_file_not_found=True)
        assert code == 1

    def test_prompt_box_falls_back_to_python_bin_dir(self):
        """shutil.which returns None but binary exists in the Python bin dir."""
        fake_python = "/fake/env/bin/python"
        expected_binary = "/fake/env/bin/prompt-box"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run = MagicMock(return_value=mock_result)

        with patch("sys.argv", ["claude-queue", "prompt-box"]):
            with patch("claude_code_queue.cli.QueueManager"):
                with patch("shutil.which", return_value=None):
                    with patch("sys.executable", fake_python):
                        with patch("claude_code_queue.cli.os.path.exists", return_value=True):
                            with patch("claude_code_queue.cli.subprocess.run", mock_run):
                                code = main()

        assert code == 0
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == expected_binary


# ===========================================================================
# `batch` Command
# ===========================================================================

def _make_mock_prompt(priority=5, content="Review file.h", working_directory="/src"):
    """Build a minimal QueuedPrompt-like mock for batch tests."""
    p = MagicMock(spec=QueuedPrompt)
    p.priority = priority
    p.content = content
    p.working_directory = working_directory
    return p


class TestBatchNoSubcommand:
    def test_batch_no_subcommand_returns_1(self, capsys):
        with patch("sys.argv", ["claude-queue", "batch"]):
            with patch("claude_code_queue.cli.QueueStorage"):
                code = main()
        assert code == 1
        assert "No batch operation specified" in capsys.readouterr().out


class TestBatchGenerate:
    def _run(self, tmp_path, *extra_args, prompts=None, resolve_side_effect=None,
             generate_side_effect=None):
        """Run `claude-queue batch generate tmpl --data data.csv [extra_args]`.

        Creates a real data file so the is_file() check passes.
        Patches resolve_template_path and generate_batch_jobs.
        """
        data_file = tmp_path / "data.csv"
        data_file.write_text("project,filename\nvolk,a.h\n")

        template_file = tmp_path / "template.md"
        template_file.write_text("---\npriority: 5\n---\n\nhello")

        mock_storage = MagicMock()
        mock_storage.bank_dir = tmp_path / "bank"

        if prompts is None:
            prompts = [_make_mock_prompt(priority=10), _make_mock_prompt(priority=15)]

        argv = [
            "claude-queue", "batch", "generate", str(template_file),
            "--data", str(data_file),
        ] + list(extra_args)

        with patch("sys.argv", argv):
            with patch("claude_code_queue.cli.QueueStorage", return_value=mock_storage):
                with patch(
                    "claude_code_queue.cli.resolve_template_path",
                    side_effect=resolve_side_effect or (lambda ref, bank: template_file),
                ):
                    with patch(
                        "claude_code_queue.cli.generate_batch_jobs",
                        side_effect=generate_side_effect or (lambda **kw: prompts),
                    ):
                        code = main()
        return code

    def test_batch_generate_success(self, tmp_path, capsys):
        code = self._run(tmp_path)
        assert code == 0
        out = capsys.readouterr().out
        assert "Generated 2 job(s)" in out

    def test_batch_generate_success_shows_priority_range(self, tmp_path, capsys):
        prompts = [_make_mock_prompt(priority=10), _make_mock_prompt(priority=15)]
        self._run(tmp_path, prompts=prompts)
        out = capsys.readouterr().out
        assert "10" in out
        assert "15" in out

    def test_batch_generate_single_job_priority_range(self, tmp_path, capsys):
        prompts = [_make_mock_prompt(priority=5)]
        self._run(tmp_path, prompts=prompts)
        out = capsys.readouterr().out
        assert "1 job(s)" in out
        # min and max are both 5
        assert out.count("5") >= 2

    def test_batch_generate_dry_run_output(self, tmp_path, capsys):
        prompts = [_make_mock_prompt(priority=7, content="Review x.h in project")]
        code = self._run(tmp_path, "--dry-run", prompts=prompts)
        assert code == 0
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "would generate" in out

    def test_batch_generate_data_file_not_found(self, tmp_path, capsys):
        """Missing data file → return 1 before touching template or generate."""
        with patch("sys.argv", [
            "claude-queue", "batch", "generate", "tmpl",
            "--data", str(tmp_path / "nonexistent.csv"),
        ]):
            with patch("claude_code_queue.cli.QueueStorage"):
                code = main()
        assert code == 1
        assert "Data file not found" in capsys.readouterr().out

    def test_batch_generate_template_not_found(self, tmp_path, capsys):
        code = self._run(
            tmp_path,
            resolve_side_effect=FileNotFoundError("Template 'tmpl' not found"),
        )
        assert code == 1
        assert "Error:" in capsys.readouterr().out

    def test_batch_generate_validation_error(self, tmp_path, capsys):
        code = self._run(
            tmp_path,
            generate_side_effect=ValueError("Template variables not found in data"),
        )
        assert code == 1
        assert "Error:" in capsys.readouterr().out


class TestBatchValidate:
    def _run(self, tmp_path, template_text, csv_text, *extra_args,
             resolve_side_effect=None):
        """Run `claude-queue batch validate tmpl --data data.csv`."""
        template_file = tmp_path / "template.md"
        template_file.write_text(template_text)

        data_file = tmp_path / "data.csv"
        data_file.write_text(csv_text)

        mock_storage = MagicMock()
        mock_storage.bank_dir = tmp_path / "bank"

        argv = [
            "claude-queue", "batch", "validate", str(template_file),
            "--data", str(data_file),
        ] + list(extra_args)

        with patch("sys.argv", argv):
            with patch("claude_code_queue.cli.QueueStorage", return_value=mock_storage):
                with patch(
                    "claude_code_queue.cli.resolve_template_path",
                    side_effect=resolve_side_effect or (lambda ref, bank: template_file),
                ):
                    code = main()
        return code

    def test_batch_validate_valid_returns_0(self, tmp_path, capsys):
        code = self._run(
            tmp_path,
            "---\npriority: 0\n---\n\nHello {{name}}",
            "name\nalice\n",
        )
        assert code == 0
        assert "Valid:" in capsys.readouterr().out

    def test_batch_validate_with_warnings_returns_0(self, tmp_path, capsys):
        code = self._run(
            tmp_path,
            "---\npriority: 0\n---\n\nHello {{name}}",
            "name,extra\nalice,unused\n",
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "Valid with warnings" in out

    def test_batch_validate_with_errors_returns_1(self, tmp_path, capsys):
        """Template references {{name}} but CSV only has 'other' column → error."""
        code = self._run(
            tmp_path,
            "---\npriority: 0\n---\n\nHello {{name}}",
            "other\nval\n",
        )
        assert code == 1
        assert "Error" in capsys.readouterr().out

    def test_batch_validate_data_file_not_found(self, tmp_path, capsys):
        template_file = tmp_path / "t.md"
        template_file.write_text("hello")
        mock_storage = MagicMock()
        mock_storage.bank_dir = tmp_path / "bank"

        with patch("sys.argv", [
            "claude-queue", "batch", "validate", str(template_file),
            "--data", str(tmp_path / "nonexistent.csv"),
        ]):
            with patch("claude_code_queue.cli.QueueStorage", return_value=mock_storage):
                code = main()
        assert code == 1
        assert "Data file not found" in capsys.readouterr().out

    def test_batch_validate_template_not_found(self, tmp_path, capsys):
        data_file = tmp_path / "data.csv"
        data_file.write_text("name\nalice\n")
        mock_storage = MagicMock()
        mock_storage.bank_dir = tmp_path / "bank"

        with patch("sys.argv", [
            "claude-queue", "batch", "validate", "nonexistent",
            "--data", str(data_file),
        ]):
            with patch("claude_code_queue.cli.QueueStorage", return_value=mock_storage):
                with patch(
                    "claude_code_queue.cli.resolve_template_path",
                    side_effect=FileNotFoundError("not found"),
                ):
                    code = main()
        assert code == 1
        assert "Error:" in capsys.readouterr().out

    def test_batch_validate_output_shows_row_count(self, tmp_path, capsys):
        self._run(
            tmp_path,
            "---\npriority: 0\n---\n\nHello {{name}}",
            "name\nalice\nbob\ncharlie\n",
        )
        out = capsys.readouterr().out
        assert "3 row(s)" in out


class TestBatchVariables:
    def _run(self, tmp_path, template_text, resolve_side_effect=None):
        template_file = tmp_path / "template.md"
        template_file.write_text(template_text)

        mock_storage = MagicMock()
        mock_storage.bank_dir = tmp_path / "bank"

        with patch("sys.argv", ["claude-queue", "batch", "variables", str(template_file)]):
            with patch("claude_code_queue.cli.QueueStorage", return_value=mock_storage):
                with patch(
                    "claude_code_queue.cli.resolve_template_path",
                    side_effect=resolve_side_effect or (lambda ref, bank: template_file),
                ):
                    code = main()
        return code

    def test_batch_variables_lists_variables(self, tmp_path, capsys):
        code = self._run(tmp_path, "---\npriority: 0\n---\n\nReview {{filename}} in {{project}}")
        assert code == 0
        out = capsys.readouterr().out
        assert "{{filename}}" in out
        assert "{{project}}" in out

    def test_batch_variables_no_variables(self, tmp_path, capsys):
        code = self._run(tmp_path, "---\npriority: 0\n---\n\nNo placeholders here")
        assert code == 0
        assert "No variables found" in capsys.readouterr().out

    def test_batch_variables_template_not_found(self, tmp_path, capsys):
        mock_storage = MagicMock()
        mock_storage.bank_dir = tmp_path / "bank"

        with patch("sys.argv", ["claude-queue", "batch", "variables", "nonexistent"]):
            with patch("claude_code_queue.cli.QueueStorage", return_value=mock_storage):
                with patch(
                    "claude_code_queue.cli.resolve_template_path",
                    side_effect=FileNotFoundError("not found"),
                ):
                    code = main()
        assert code == 1
        assert "Error:" in capsys.readouterr().out

    def test_batch_variables_returns_0(self, tmp_path):
        code = self._run(tmp_path, "---\npriority: 0\n---\n\nProcess {{item}}")
        assert code == 0


# ===========================================================================
# Cleanup Command
# ===========================================================================

class TestCleanup:
    """Tests for `claude-queue cleanup [--dry-run]`."""

    def _make_artifacts(self, tmp_path, session_uuid="aaa-bbb-ccc"):
        """Create fake rate-limit artifacts under tmp_path/.claude/."""
        claude_dir = tmp_path / ".claude"
        debug_dir = claude_dir / "debug"
        projects_dir = claude_dir / "projects" / "-home-testuser-project"
        todos_dir = claude_dir / "todos"
        telemetry_dir = claude_dir / "telemetry"
        for d in (debug_dir, projects_dir, todos_dir, telemetry_dir):
            d.mkdir(parents=True)

        # Debug file with rate_limit_error content
        debug_file = debug_dir / f"{session_uuid}.txt"
        debug_file.write_text("startup\nrate_limit_error\n")

        # Correlated JSONL
        jsonl_file = projects_dir / f"{session_uuid}.jsonl"
        jsonl_file.write_bytes(b"x" * 5000)

        # Correlated todo stub
        todo_file = todos_dir / f"{session_uuid}-agent-{session_uuid}.json"
        todo_file.write_text("[]")

        # Correlated telemetry file
        telemetry_file = telemetry_dir / f"1p_failed_events.{session_uuid}.other-uuid.json"
        telemetry_file.write_text('{"events": []}')

        return debug_file, jsonl_file, todo_file, telemetry_file

    def test_cleanup_dry_run_does_not_delete(self, tmp_path, capsys):
        debug_file, jsonl_file, todo_file, telemetry_file = self._make_artifacts(tmp_path)

        with patch("sys.argv", ["claude-queue", "cleanup", "--dry-run"]):
            with patch("pathlib.Path.home", return_value=tmp_path):
                code = main()

        assert code == 0
        assert debug_file.exists(), "dry-run must not delete files"
        assert jsonl_file.exists()
        assert todo_file.exists()
        assert telemetry_file.exists()
        out = capsys.readouterr().out
        assert "Would delete" in out
        assert "dry-run" in out

    def test_cleanup_deletes_artifacts(self, tmp_path, capsys):
        debug_file, jsonl_file, todo_file, telemetry_file = self._make_artifacts(tmp_path)

        with patch("sys.argv", ["claude-queue", "cleanup"]):
            with patch("pathlib.Path.home", return_value=tmp_path):
                code = main()

        assert code == 0
        assert not debug_file.exists()
        assert not jsonl_file.exists()
        assert not todo_file.exists()
        assert not telemetry_file.exists()
        out = capsys.readouterr().out
        assert "Deleted 4 rate-limit artifact(s)" in out

    def test_cleanup_preserves_non_rate_limited_debug(self, tmp_path, capsys):
        """Debug files without rate_limit_error are not deleted."""
        claude_dir = tmp_path / ".claude"
        debug_dir = claude_dir / "debug"
        debug_dir.mkdir(parents=True)

        good_file = debug_dir / "good-session.txt"
        good_file.write_text("startup\nall good\nstream completed\n")

        with patch("sys.argv", ["claude-queue", "cleanup"]):
            with patch("pathlib.Path.home", return_value=tmp_path):
                code = main()

        assert code == 0
        assert good_file.exists()
        assert "Deleted 0" in capsys.readouterr().out

    def test_cleanup_preserves_real_todo_file(self, tmp_path, capsys):
        """Todo files > 2 bytes are preserved even if UUID matches a rate-limited session."""
        debug_file, jsonl_file, todo_file, telemetry_file = self._make_artifacts(tmp_path)
        # Overwrite the stub with realistic todo content (> 2 bytes)
        todo_file.write_text('[{"task": "implement feature", "status": "in_progress"}]')

        with patch("sys.argv", ["claude-queue", "cleanup"]):
            with patch("pathlib.Path.home", return_value=tmp_path):
                code = main()

        assert code == 0
        assert todo_file.exists(), "real todo file (> 2 bytes) must be preserved"
        assert not debug_file.exists()
        assert not jsonl_file.exists()
        # 3 deleted: debug + jsonl + telemetry (todo preserved by size guard)
        assert "Deleted 3" in capsys.readouterr().out

    def test_cleanup_handles_empty_claude_dir(self, tmp_path, capsys):
        """Cleanup succeeds when ~/.claude/ has no artifact directories."""
        (tmp_path / ".claude").mkdir()

        with patch("sys.argv", ["claude-queue", "cleanup"]):
            with patch("pathlib.Path.home", return_value=tmp_path):
                code = main()

        assert code == 0
        assert "Deleted 0" in capsys.readouterr().out

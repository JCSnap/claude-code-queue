"""
Tests for the CLI module — Pass 7.

Covers: argument parsing, global option forwarding, command dispatch,
output format (text and JSON), bank sub-commands, prompt-box passthrough,
and exit codes for every command.

All tests mock ``QueueManager`` to avoid real storage / subprocess calls.
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
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.add_prompt.return_value = success
                mock_cls.return_value = mgr
                code = main()
                return code, mgr

    def test_add_prompt_content(self):
        _, mgr = self._run_add()
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.content == "hello world"

    def test_add_default_priority_zero(self):
        _, mgr = self._run_add()
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.priority == 0

    def test_add_long_priority_flag(self):
        _, mgr = self._run_add("--priority", "5")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.priority == 5

    def test_add_short_priority_flag(self):
        _, mgr = self._run_add("-p", "3")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.priority == 3

    def test_add_default_working_dir_dot(self):
        _, mgr = self._run_add()
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.working_directory == "."

    def test_add_long_working_dir_flag(self):
        _, mgr = self._run_add("--working-dir", "/some/path")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.working_directory == "/some/path"

    def test_add_short_working_dir_flag(self):
        _, mgr = self._run_add("-d", "/some/path")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.working_directory == "/some/path"

    def test_add_context_files_long_flag(self):
        _, mgr = self._run_add("--context-files", "a.py", "b.py")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.context_files == ["a.py", "b.py"]

    def test_add_context_files_short_flag(self):
        _, mgr = self._run_add("-f", "a.py")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.context_files == ["a.py"]

    def test_add_default_context_files_empty(self):
        _, mgr = self._run_add()
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.context_files == []

    def test_add_max_retries_long_flag(self):
        _, mgr = self._run_add("--max-retries", "5")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.max_retries == 5

    def test_add_max_retries_short_flag(self):
        _, mgr = self._run_add("-r", "2")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.max_retries == 2

    def test_add_default_max_retries_three(self):
        _, mgr = self._run_add()
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.max_retries == 3

    def test_add_estimated_tokens_long_flag(self):
        _, mgr = self._run_add("--estimated-tokens", "1500")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.estimated_tokens == 1500

    def test_add_estimated_tokens_short_flag(self):
        _, mgr = self._run_add("-t", "1000")
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.estimated_tokens == 1000

    def test_add_default_estimated_tokens_none(self):
        _, mgr = self._run_add()
        prompt = mgr.add_prompt.call_args[0][0]
        assert prompt.estimated_tokens is None

    def test_add_returns_zero_on_success(self):
        code, _ = self._run_add(success=True)
        assert code == 0

    def test_add_returns_one_on_failure(self):
        code, _ = self._run_add(success=False)
        assert code == 1


# ===========================================================================
# `template` Command
# ===========================================================================

class TestTemplateCommand:
    def _run_template(self, *extra_args, path="/some/path/my-template.md"):
        with patch("sys.argv", ["claude-queue", "template", "my-template"] + list(extra_args)):
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.create_prompt_template.return_value = path
                mock_cls.return_value = mgr
                code = main()
                return code, mgr

    def test_template_calls_create_with_filename(self):
        _, mgr = self._run_template()
        mgr.create_prompt_template.assert_called_once_with("my-template", 0)

    def test_template_default_priority_zero(self):
        _, mgr = self._run_template()
        _, priority = mgr.create_prompt_template.call_args[0]
        assert priority == 0

    def test_template_long_priority_flag(self):
        _, mgr = self._run_template("--priority", "2")
        _, priority = mgr.create_prompt_template.call_args[0]
        assert priority == 2

    def test_template_short_priority_flag(self):
        _, mgr = self._run_template("-p", "1")
        _, priority = mgr.create_prompt_template.call_args[0]
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
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.get_status.return_value = state
                mock_cls.return_value = mgr
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
        with patch("sys.argv", ["claude-queue", "cancel", prompt_id]):
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.remove_prompt.return_value = success
                mock_cls.return_value = mgr
                code = main()
                return code, mgr

    def test_cancel_calls_remove_prompt_with_id(self):
        _, mgr = self._run_cancel("abc123")
        mgr.remove_prompt.assert_called_once_with("abc123")

    def test_cancel_returns_zero_on_success(self):
        code, _ = self._run_cancel("abc123", success=True)
        assert code == 0

    def test_cancel_returns_one_on_failure(self):
        code, _ = self._run_cancel("abc123", success=False)
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
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.get_status.return_value = state
                mock_cls.return_value = mgr
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
                with patch("claude_code_queue.cli.QueueManager"):
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
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.save_prompt_to_bank.return_value = self._BANK_PATH
                mock_cls.return_value = mgr
                code = main()
                return code, mgr

    def test_bank_save_calls_save_prompt_to_bank(self):
        _, mgr = self._run_bank_save()
        mgr.save_prompt_to_bank.assert_called_once_with("my-template", 0)

    def test_bank_save_default_priority_zero(self):
        _, mgr = self._run_bank_save()
        _, priority = mgr.save_prompt_to_bank.call_args[0]
        assert priority == 0

    def test_bank_save_long_priority_flag(self):
        _, mgr = self._run_bank_save("--priority", "3")
        _, priority = mgr.save_prompt_to_bank.call_args[0]
        assert priority == 3

    def test_bank_save_short_priority_flag(self):
        _, mgr = self._run_bank_save("-p", "2")
        _, priority = mgr.save_prompt_to_bank.call_args[0]
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
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.list_bank_templates.return_value = templates
                mock_cls.return_value = mgr
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
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.use_bank_template.return_value = success
                mock_cls.return_value = mgr
                return main(), mgr

    def test_bank_use_calls_use_bank_template(self):
        _, mgr = self._run_bank_use()
        mgr.use_bank_template.assert_called_once_with("my-template")

    def test_bank_use_returns_zero_on_success(self):
        code, _ = self._run_bank_use(success=True)
        assert code == 0

    def test_bank_use_returns_one_on_failure(self):
        code, _ = self._run_bank_use(success=False)
        assert code == 1


# ===========================================================================
# `bank delete` Command
# ===========================================================================

class TestBankDeleteCommand:
    def _run_bank_delete(self, success=True):
        with patch("sys.argv", ["claude-queue", "bank", "delete", "my-template"]):
            with patch("claude_code_queue.cli.QueueManager") as mock_cls:
                mgr = MagicMock()
                mgr.delete_bank_template.return_value = success
                mock_cls.return_value = mgr
                return main(), mgr

    def test_bank_delete_calls_delete_bank_template(self):
        _, mgr = self._run_bank_delete()
        mgr.delete_bank_template.assert_called_once_with("my-template")

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
    cmd_prompt_box does not use the manager, but QueueManager is still
    instantiated in main() before the command dispatch, so we always mock it.
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

"""
Tests for rate-limit artifact cleanup in queue_manager.py.

Covers _cleanup_rate_limit_artifacts() and _do_cleanup_rate_limit_artifacts().
Uses tmp_path-based fake ~/.claude/ directories to avoid touching real state.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from claude_code_queue.models import (
    ExecutionResult,
    PromptStatus,
    QueuedPrompt,
    QueueState,
    RateLimitInfo,
)
from claude_code_queue.queue_manager import QueueManager
from claude_code_queue.claude_interface import ClaudeCodeInterface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SESSION_UUID = "00134021-1e30-4928-b9af-e92a676ab248"
OTHER_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _make_claude_dirs(tmp_path, working_dir="/home/user/src/volk"):
    """Create the four artifact directories under a fake ~/.claude/.

    Returns (claude_dir, jsonl_dir, todos_dir, debug_dir, telemetry_dir).
    """
    claude_dir = tmp_path / ".claude"
    encoded = working_dir.replace("/", "-")
    jsonl_dir = claude_dir / "projects" / encoded
    todos_dir = claude_dir / "todos"
    debug_dir = claude_dir / "debug"
    telemetry_dir = claude_dir / "telemetry"
    for d in (jsonl_dir, todos_dir, debug_dir, telemetry_dir):
        d.mkdir(parents=True)
    return claude_dir, jsonl_dir, todos_dir, debug_dir, telemetry_dir


def _write_file(path, size_bytes=100, content=None):
    """Write a file with a given size or explicit content."""
    if content is not None:
        path.write_text(content)
    else:
        path.write_bytes(b"x" * size_bytes)


def _make_prompt(working_dir="/home/user/src/volk"):
    """Create a prompt with last_executed set to 1 second ago and resolved working dir.

    Using a 1-second offset avoids filesystem mtime-precision races: ext4 has
    1-second granularity, so a file written "now" may have an mtime equal to
    or slightly before datetime.now().timestamp().
    """
    p = QueuedPrompt(
        id="abc12345",
        content="test task",
        working_directory=working_dir,
        status=PromptStatus.EXECUTING,
    )
    p.last_executed = datetime.now() - timedelta(seconds=1)
    p._resolved_working_directory = str(Path(working_dir).resolve())
    return p


def _rate_limit_result() -> ExecutionResult:
    return ExecutionResult(
        success=False,
        output="usage limit reached",
        error="",
        rate_limit_info=RateLimitInfo(is_rate_limited=True, reset_time=None),
        execution_time=0.1,
    )


# ===========================================================================
# Basic Cleanup — All Four Artifact Types  (CLN-001 through CLN-004)
# ===========================================================================


def test_cleanup_deletes_rate_limited_jsonl(tmp_path, manager):  # CLN-001
    """A small, recent JSONL file is deleted; its UUID is used for correlated cleanup."""
    claude_dir, jsonl_dir, *_ = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    jsonl_file = jsonl_dir / f"{SESSION_UUID}.jsonl"
    _write_file(jsonl_file, size_bytes=4000)  # 4 KB — rate-limited size

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert not jsonl_file.exists(), "Rate-limited JSONL file should be deleted"
    assert "Cleaned up" in prompt.execution_log


def test_cleanup_deletes_correlated_todo_stub(tmp_path, manager):  # CLN-002
    """A 2-byte todo stub whose UUID matches the JSONL file is deleted."""
    claude_dir, jsonl_dir, todos_dir, *_ = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    _write_file(jsonl_dir / f"{SESSION_UUID}.jsonl", size_bytes=4000)
    todo_file = todos_dir / f"{SESSION_UUID}-agent-{SESSION_UUID}.json"
    _write_file(todo_file, content="[]")

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert not todo_file.exists(), "2-byte todo stub should be deleted"


def test_cleanup_deletes_correlated_debug_file(tmp_path, manager):  # CLN-003
    """A debug file whose UUID matches the JSONL file is deleted."""
    claude_dir, jsonl_dir, todos_dir, debug_dir, _ = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    _write_file(jsonl_dir / f"{SESSION_UUID}.jsonl", size_bytes=4000)
    debug_file = debug_dir / f"{SESSION_UUID}.txt"
    _write_file(debug_file, size_bytes=13000)

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert not debug_file.exists(), "Correlated debug file should be deleted"


def test_cleanup_deletes_correlated_telemetry_file(tmp_path, manager):  # CLN-004
    """A telemetry file whose session UUID matches the JSONL file is deleted."""
    claude_dir, jsonl_dir, _, _, telemetry_dir = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    _write_file(jsonl_dir / f"{SESSION_UUID}.jsonl", size_bytes=4000)
    telemetry_file = telemetry_dir / f"1p_failed_events.{SESSION_UUID}.{OTHER_UUID}.json"
    _write_file(telemetry_file, size_bytes=30000)

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert not telemetry_file.exists(), "Correlated telemetry file should be deleted"


# ===========================================================================
# Preservation — Files That Must NOT Be Deleted  (CLN-005 through CLN-009)
# ===========================================================================


def test_cleanup_preserves_old_jsonl(tmp_path, manager):  # CLN-005
    """JSONL files older than last_executed are not deleted."""
    claude_dir, jsonl_dir, *_ = _make_claude_dirs(tmp_path)

    old_jsonl = jsonl_dir / f"{SESSION_UUID}.jsonl"
    _write_file(old_jsonl, size_bytes=4000)
    # Set mtime to 1 hour ago
    old_time = time.time() - 3600
    os.utime(old_jsonl, (old_time, old_time))

    prompt = _make_prompt()

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert old_jsonl.exists(), "Old JSONL file must be preserved"


def test_cleanup_preserves_large_jsonl(tmp_path, manager):  # CLN-006
    """JSONL files >= 10 KB (successful runs) are not deleted even if recent."""
    claude_dir, jsonl_dir, *_ = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    large_jsonl = jsonl_dir / f"{SESSION_UUID}.jsonl"
    _write_file(large_jsonl, size_bytes=150_000)  # 150 KB — successful run

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert large_jsonl.exists(), "Large JSONL file (successful run) must be preserved"


def test_cleanup_preserves_non_empty_todo(tmp_path, manager):  # CLN-007
    """Todo files > 2 bytes are not deleted even when UUID-correlated."""
    claude_dir, jsonl_dir, todos_dir, *_ = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    _write_file(jsonl_dir / f"{SESSION_UUID}.jsonl", size_bytes=4000)
    real_todo = todos_dir / f"{SESSION_UUID}-agent-{SESSION_UUID}.json"
    _write_file(real_todo, size_bytes=800)  # legitimate todo

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert real_todo.exists(), "Non-empty todo file must be preserved"


def test_cleanup_preserves_old_debug_file(tmp_path, manager):  # CLN-008
    """Debug files older than last_executed are not deleted even with UUID match."""
    claude_dir, jsonl_dir, _, debug_dir, _ = _make_claude_dirs(tmp_path)

    # Create JSONL file with current timestamp
    _write_file(jsonl_dir / f"{SESSION_UUID}.jsonl", size_bytes=4000)

    # Create debug file with old timestamp
    debug_file = debug_dir / f"{SESSION_UUID}.txt"
    _write_file(debug_file, size_bytes=13000)
    old_time = time.time() - 3600
    os.utime(debug_file, (old_time, old_time))

    prompt = _make_prompt()

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert debug_file.exists(), "Old debug file must be preserved (timestamp guard)"


def test_cleanup_does_not_delete_debug_without_jsonl_match(tmp_path, manager):  # CLN-009
    """Debug files are only deleted when their UUID matches a rate-limited JSONL file.

    If no JSONL file matches (e.g. it's >= 10 KB), the debug file is untouched.
    """
    claude_dir, jsonl_dir, _, debug_dir, _ = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    # JSONL file is large (successful run) — no UUID collected
    _write_file(jsonl_dir / f"{SESSION_UUID}.jsonl", size_bytes=150_000)

    # Debug file exists for same UUID — must NOT be deleted
    debug_file = debug_dir / f"{SESSION_UUID}.txt"
    _write_file(debug_file, size_bytes=13000)

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert debug_file.exists(), "Debug file must not be deleted without JSONL UUID match"


# ===========================================================================
# Cleanup Not Triggered for Other Result Types  (CLN-010, CLN-011)
# ===========================================================================


def test_cleanup_not_called_on_success(tmp_path, manager, mocker):  # CLN-010
    """Successful execution does not trigger artifact cleanup."""
    prompt = QueuedPrompt(content="task")
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    spy = mocker.patch.object(manager, "_cleanup_rate_limit_artifacts")

    success = ExecutionResult(success=True, output="done", error="", execution_time=0.1)
    mocker.patch.object(manager.claude_interface, "execute_prompt", return_value=success)
    manager._execute_prompt(prompt)

    spy.assert_not_called()


def test_cleanup_not_called_on_generic_failure(tmp_path, manager, mocker):  # CLN-011
    """Generic failure does not trigger artifact cleanup."""
    prompt = QueuedPrompt(content="task", max_retries=3)
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    spy = mocker.patch.object(manager, "_cleanup_rate_limit_artifacts")

    fail = ExecutionResult(success=False, output="", error="oops", execution_time=0.1)
    mocker.patch.object(manager.claude_interface, "execute_prompt", return_value=fail)
    manager._execute_prompt(prompt)

    spy.assert_not_called()


def test_cleanup_called_on_rate_limit(tmp_path, manager, mocker):  # CLN-012
    """Rate-limited execution triggers artifact cleanup."""
    prompt = QueuedPrompt(content="task", max_retries=3)
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    spy = mocker.patch.object(manager, "_cleanup_rate_limit_artifacts")

    mocker.patch.object(
        manager.claude_interface, "execute_prompt", return_value=_rate_limit_result()
    )
    manager._execute_prompt(prompt)

    spy.assert_called_once_with(prompt)


# ===========================================================================
# Missing Directories  (CLN-013)
# ===========================================================================


def test_cleanup_handles_missing_directories(tmp_path, manager):  # CLN-013
    """Cleanup does not crash when artifact directories don't exist."""
    # Point home at tmp_path which has no .claude/ at all
    prompt = _make_prompt()
    prompt.last_executed = datetime.now()

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        # Should not raise
        manager._cleanup_rate_limit_artifacts(prompt)

    # No log entry since nothing was deleted
    assert "Cleaned up" not in prompt.execution_log


# ===========================================================================
# Per-File Exception Handling  (CLN-014)
# ===========================================================================


def test_cleanup_continues_after_oserror_on_one_file(tmp_path, manager, mocker):  # CLN-014
    """If stat() raises OSError on the debug file, the todo file is still deleted."""
    _, jsonl_dir, todos_dir, debug_dir, _ = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    # Create one small JSONL
    _write_file(jsonl_dir / f"{SESSION_UUID}.jsonl", size_bytes=4000)

    # Create debug file normally
    debug_file = debug_dir / f"{SESSION_UUID}.txt"
    _write_file(debug_file, size_bytes=13000)

    # Create todo stub — should still be cleaned up despite debug failure
    todo_file = todos_dir / f"{SESSION_UUID}-agent-{SESSION_UUID}.json"
    _write_file(todo_file, content="[]")

    # Patch Path.stat to raise OSError only for the debug file
    original_stat = Path.stat

    def selective_stat(self, *args, **kwargs):
        if str(self) == str(debug_file):
            raise OSError("Permission denied")
        return original_stat(self, *args, **kwargs)

    mocker.patch.object(Path, "stat", selective_stat)

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    # Stop all mocks before checking file existence (exists() calls stat())
    mocker.stopall()
    assert not todo_file.exists(), "Todo file should still be deleted despite debug OSError"
    assert debug_file.exists(), "Debug file should survive (stat raised OSError)"


# ===========================================================================
# Top-Level Exception Safety  (CLN-015)
# ===========================================================================


def test_cleanup_exception_does_not_break_result_processing(tmp_path, manager, mocker):  # CLN-015
    """If the entire cleanup throws, _process_execution_result() still completes
    and the prompt's RATE_LIMITED status is persisted.
    """
    prompt = QueuedPrompt(content="task", max_retries=3)
    manager.state = QueueState()
    manager.state.add_prompt(prompt)

    # Make cleanup explode
    mocker.patch.object(
        manager, "_do_cleanup_rate_limit_artifacts",
        side_effect=RuntimeError("disk on fire")
    )

    rl_result = _rate_limit_result()
    prompt.status = PromptStatus.EXECUTING
    prompt.last_executed = datetime.now()
    manager._process_execution_result(prompt, rl_result)

    assert prompt.status == PromptStatus.RATE_LIMITED, (
        "Prompt must reach RATE_LIMITED status even when cleanup throws"
    )
    assert manager.state.last_processed is not None, (
        "last_processed must be set even when cleanup throws"
    )
    assert "artifact cleanup failed" in prompt.execution_log


# ===========================================================================
# No last_executed Guard  (CLN-016)
# ===========================================================================


def test_cleanup_noop_without_last_executed(manager):  # CLN-016
    """Cleanup is a no-op when prompt.last_executed is None."""
    prompt = QueuedPrompt(content="task")
    prompt.last_executed = None

    manager.state = QueueState()
    manager.state.add_prompt(prompt)

    # Should not raise and should not log
    manager._cleanup_rate_limit_artifacts(prompt)
    assert "Cleaned up" not in prompt.execution_log


# ===========================================================================
# Resolved Working Directory  (CLN-017, CLN-018)
# ===========================================================================


def test_execute_prompt_stashes_resolved_working_directory(manager, mocker):  # CLN-017
    """_execute_prompt() sets _resolved_working_directory on the prompt."""
    prompt = QueuedPrompt(content="task", working_directory="/some/path")
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    mocker.patch.object(
        manager.claude_interface, "execute_prompt",
        return_value=ExecutionResult(success=True, output="ok", error="", execution_time=0.1),
    )
    manager._execute_prompt(prompt)

    assert prompt._resolved_working_directory is not None
    assert prompt._resolved_working_directory == str(Path("/some/path").resolve())


def test_cleanup_uses_resolved_working_directory(tmp_path, manager):  # CLN-018
    """Cleanup uses _resolved_working_directory (not re-resolving working_directory)."""
    # Set up dirs for the resolved path, not the relative one
    real_path = "/home/user/src/volk"
    claude_dir, jsonl_dir, *_ = _make_claude_dirs(tmp_path, working_dir=real_path)

    prompt = QueuedPrompt(
        content="task",
        working_directory=".",  # relative — would resolve to CWD
        status=PromptStatus.EXECUTING,
    )
    prompt.last_executed = datetime.now() - timedelta(seconds=1)
    prompt._resolved_working_directory = real_path  # stashed at execution time

    jsonl_file = jsonl_dir / f"{SESSION_UUID}.jsonl"
    _write_file(jsonl_file, size_bytes=4000)

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert not jsonl_file.exists(), (
        "Cleanup must use _resolved_working_directory, not re-resolve '.'"
    )


# ===========================================================================
# Deleted Count and Logging  (CLN-019)
# ===========================================================================


def test_cleanup_counts_all_deleted_artifacts(tmp_path, manager, capsys):  # CLN-019
    """The deleted count includes all four artifact types."""
    claude_dir, jsonl_dir, todos_dir, debug_dir, telemetry_dir = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    _write_file(jsonl_dir / f"{SESSION_UUID}.jsonl", size_bytes=4000)
    _write_file(todos_dir / f"{SESSION_UUID}-agent-{SESSION_UUID}.json", content="[]")
    _write_file(debug_dir / f"{SESSION_UUID}.txt", size_bytes=13000)
    _write_file(telemetry_dir / f"1p_failed_events.{SESSION_UUID}.{OTHER_UUID}.json", size_bytes=30000)

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert "Cleaned up 4 rate-limit artifact(s)" in prompt.execution_log
    captured = capsys.readouterr()
    assert "[cleanup] Removed 4 rate-limit artifact(s)" in captured.out


def test_cleanup_no_log_when_nothing_deleted(tmp_path, manager, capsys):  # CLN-020
    """No log entry or print when zero files are deleted."""
    _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert "Cleaned up" not in prompt.execution_log
    captured = capsys.readouterr()
    assert "[cleanup]" not in captured.out


# ===========================================================================
# JSONL Early Break  (CLN-021)
# ===========================================================================


def test_cleanup_breaks_after_first_jsonl_match(tmp_path, manager):  # CLN-021
    """Only one JSONL file is deleted per cleanup (one subprocess = one UUID).

    Even with multiple small recent JSONL files, only the first match is deleted.
    """
    uuid2 = "99999999-aaaa-bbbb-cccc-dddddddddddd"
    claude_dir, jsonl_dir, *_ = _make_claude_dirs(tmp_path)
    prompt = _make_prompt()

    f1 = jsonl_dir / f"{SESSION_UUID}.jsonl"
    f2 = jsonl_dir / f"{uuid2}.jsonl"
    _write_file(f1, size_bytes=4000)
    _write_file(f2, size_bytes=4000)

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    # Exactly one should be deleted (we don't know which due to glob ordering)
    remaining = list(jsonl_dir.glob("*.jsonl"))
    assert len(remaining) == 1, (
        f"Expected exactly 1 JSONL file remaining after cleanup, got {len(remaining)}"
    )


# ===========================================================================
# _resolved_working_directory Field  (CLN-022)
# ===========================================================================


def test_resolved_working_directory_not_persisted_to_yaml(tmp_path, manager):  # CLN-022
    """_resolved_working_directory is transient and not written to YAML frontmatter."""
    prompt = QueuedPrompt(content="task", working_directory="/some/path")
    prompt._resolved_working_directory = "/some/path"
    prompt.last_executed = datetime.now()

    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)
    manager.storage.save_queue_state(manager.state)

    # Read the file back and check YAML doesn't contain the field
    queue_files = list(manager.storage.queue_dir.glob("*.md"))
    assert len(queue_files) == 1
    content = queue_files[0].read_text()
    assert "_resolved_working_directory" not in content

    # Reload and verify it's None (not persisted)
    reloaded = manager.storage.load_queue_state()
    reloaded_prompt = reloaded.prompts[0]
    assert reloaded_prompt._resolved_working_directory is None


# ===========================================================================
# Fallback When _resolved_working_directory Is None  (CLN-023)
# ===========================================================================


def test_cleanup_falls_back_to_resolve_when_stash_missing(tmp_path, manager):  # CLN-023
    """If _resolved_working_directory is None, cleanup resolves working_directory directly."""
    working_dir = "/home/user/src/volk"
    claude_dir, jsonl_dir, *_ = _make_claude_dirs(tmp_path, working_dir=working_dir)
    prompt = QueuedPrompt(
        content="task",
        working_directory=working_dir,
        status=PromptStatus.EXECUTING,
    )
    prompt.last_executed = datetime.now() - timedelta(seconds=1)
    prompt._resolved_working_directory = None  # simulate missing stash

    jsonl_file = jsonl_dir / f"{SESSION_UUID}.jsonl"
    _write_file(jsonl_file, size_bytes=4000)

    with patch("pathlib.Path.home", return_value=tmp_path):
        manager.state = QueueState()
        manager.state.add_prompt(prompt)
        manager._cleanup_rate_limit_artifacts(prompt)

    assert not jsonl_file.exists(), "Cleanup should fall back to resolving working_directory"

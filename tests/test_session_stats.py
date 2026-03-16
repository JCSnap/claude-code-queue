"""
Tests for SessionStats dataclass and session stats extraction from JSONL logs.

Test IDs use the SS- prefix for cross-reference.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_code_queue.models import (
    SessionStats,
    QueuedPrompt,
    PromptStatus,
    ExecutionResult,
    RateLimitInfo,
)
from claude_code_queue.queue_manager import QueueManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant_line(
    input_tokens=10,
    output_tokens=20,
    cache_creation=100,
    cache_read=200,
):
    """Build a single JSONL assistant line with the given usage values."""
    return json.dumps({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
        },
    })


def _make_user_line():
    """Build a JSONL user line (should be ignored by stats extraction)."""
    return json.dumps({
        "type": "user",
        "message": {"role": "user", "content": "say hello"},
    })


def _make_queue_op_line():
    """Build a JSONL queue-operation line (should be ignored)."""
    return json.dumps({
        "type": "queue-operation",
        "operation": "enqueue",
        "timestamp": "2026-03-15T12:00:00.000Z",
    })


def _make_last_prompt_line():
    """Build a JSONL last-prompt line (should be ignored)."""
    return json.dumps({
        "type": "last-prompt",
        "lastPrompt": "say hello",
    })


def _write_jsonl(path, lines):
    """Write JSONL lines to a file and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    return path


def _setup_jsonl_for_prompt(tmp_path, prompt, lines):
    """Create the JSONL file in the expected directory structure for a prompt.

    Returns the path to the JSONL file.
    """
    resolved = prompt._resolved_working_directory or str(
        Path(prompt.working_directory).resolve()
    )
    encoded = resolved.replace("/", "-")
    jsonl_dir = tmp_path / ".claude" / "projects" / encoded
    jsonl_file = jsonl_dir / "session-uuid.jsonl"
    _write_jsonl(jsonl_file, lines)
    return jsonl_file


def _make_stats_prompt(tmp_path):
    """Create a QueuedPrompt wired to a working directory under tmp_path."""
    work_dir = tmp_path / "workdir"
    work_dir.mkdir(exist_ok=True)
    prompt = QueuedPrompt(
        id="abc12345",
        content="test",
        working_directory=str(work_dir),
    )
    prompt.last_executed = datetime.now() - timedelta(seconds=5)
    prompt._resolved_working_directory = str(work_dir)
    return prompt


# ===========================================================================
# SessionStats — basic properties
# ===========================================================================


def test_session_stats_defaults_are_zero():  # SS-001
    stats = SessionStats()
    assert stats.input_tokens == 0
    assert stats.output_tokens == 0
    assert stats.cache_creation_input_tokens == 0
    assert stats.cache_read_input_tokens == 0
    assert stats.api_turns == 0


def test_session_stats_total_input_sums_all_three():  # SS-002
    stats = SessionStats(
        input_tokens=10,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=200,
    )
    assert stats.total_input_tokens == 310


def test_session_stats_total_input_zero_when_all_zero():  # SS-003
    stats = SessionStats()
    assert stats.total_input_tokens == 0


# ===========================================================================
# _extract_session_stats()
# ===========================================================================


def test_extract_stats_single_turn(manager, tmp_path, mocker):  # SS-010
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    jsonl_file = _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_user_line(),
        _make_assistant_line(input_tokens=5, output_tokens=50, cache_creation=1000, cache_read=2000),
    ])
    os.utime(jsonl_file, (datetime.now().timestamp(), datetime.now().timestamp()))

    stats = manager._extract_session_stats(prompt)

    assert stats is not None
    assert stats.input_tokens == 5
    assert stats.output_tokens == 50
    assert stats.cache_creation_input_tokens == 1000
    assert stats.cache_read_input_tokens == 2000
    assert stats.total_input_tokens == 3005
    assert stats.api_turns == 1


def test_extract_stats_multi_turn(manager, tmp_path, mocker):  # SS-011
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    jsonl_file = _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_user_line(),
        _make_assistant_line(input_tokens=3, output_tokens=100, cache_creation=5000, cache_read=8000),
        _make_user_line(),
        _make_assistant_line(input_tokens=1, output_tokens=200, cache_creation=5000, cache_read=8000),
        _make_user_line(),
        _make_assistant_line(input_tokens=1, output_tokens=150, cache_creation=0, cache_read=10000),
    ])
    os.utime(jsonl_file, (datetime.now().timestamp(), datetime.now().timestamp()))

    stats = manager._extract_session_stats(prompt)

    assert stats is not None
    assert stats.input_tokens == 5
    assert stats.output_tokens == 450
    assert stats.cache_creation_input_tokens == 10000
    assert stats.cache_read_input_tokens == 26000
    assert stats.total_input_tokens == 36005
    assert stats.api_turns == 3


def test_extract_stats_non_assistant_lines_ignored(manager, tmp_path, mocker):  # SS-012
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    jsonl_file = _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_queue_op_line(),
        _make_user_line(),
        _make_assistant_line(input_tokens=3, output_tokens=10, cache_creation=100, cache_read=200),
        _make_last_prompt_line(),
    ])
    os.utime(jsonl_file, (datetime.now().timestamp(), datetime.now().timestamp()))

    stats = manager._extract_session_stats(prompt)

    assert stats is not None
    assert stats.input_tokens == 3
    assert stats.output_tokens == 10
    assert stats.api_turns == 1


def test_extract_stats_missing_usage_block(manager, tmp_path, mocker):  # SS-013
    """Assistant line without message.usage should contribute 0."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    line_no_usage = json.dumps({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
        },
    })
    jsonl_file = _setup_jsonl_for_prompt(tmp_path, prompt, [
        line_no_usage,
        _make_assistant_line(input_tokens=5, output_tokens=10, cache_creation=100, cache_read=200),
    ])
    os.utime(jsonl_file, (datetime.now().timestamp(), datetime.now().timestamp()))

    stats = manager._extract_session_stats(prompt)

    assert stats is not None
    assert stats.api_turns == 2
    assert stats.input_tokens == 5
    assert stats.output_tokens == 10


def test_extract_stats_malformed_line_skipped(manager, tmp_path, mocker):  # SS-014
    """Non-JSON lines should be skipped; valid lines still counted."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    jsonl_file = _setup_jsonl_for_prompt(tmp_path, prompt, [
        "this is not json",
        _make_assistant_line(input_tokens=7, output_tokens=30, cache_creation=500, cache_read=600),
        "{bad json",
    ])
    os.utime(jsonl_file, (datetime.now().timestamp(), datetime.now().timestamp()))

    stats = manager._extract_session_stats(prompt)

    assert stats is not None
    assert stats.input_tokens == 7
    assert stats.output_tokens == 30
    assert stats.api_turns == 1


def test_extract_stats_old_mtime_returns_none(manager, tmp_path, mocker):  # SS-015
    """JSONL file exists but mtime is before cutoff — returns None."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    jsonl_file = _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_assistant_line(),
    ])
    old_time = (prompt.last_executed - timedelta(hours=1)).timestamp()
    os.utime(jsonl_file, (old_time, old_time))

    stats = manager._extract_session_stats(prompt)
    assert stats is None


def test_extract_stats_empty_file(manager, tmp_path, mocker):  # SS-016
    """Empty JSONL file — returns None (0 API turns)."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    jsonl_file = _setup_jsonl_for_prompt(tmp_path, prompt, [])
    os.utime(jsonl_file, (datetime.now().timestamp(), datetime.now().timestamp()))

    stats = manager._extract_session_stats(prompt)
    assert stats is None


def test_extract_stats_directory_missing(manager, tmp_path, mocker):  # SS-017
    """~/.claude/projects/<encoded>/ doesn't exist — returns None."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)

    stats = manager._extract_session_stats(prompt)
    assert stats is None


def test_extract_stats_resolved_dir_none_fallback(manager, tmp_path, mocker):  # SS-018
    """When _resolved_working_directory is None, falls back to resolving working_directory."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    prompt._resolved_working_directory = None
    resolved = str(Path(prompt.working_directory).resolve())
    encoded = resolved.replace("/", "-")
    jsonl_dir = tmp_path / ".claude" / "projects" / encoded
    jsonl_file = jsonl_dir / "session.jsonl"
    _write_jsonl(jsonl_file, [
        _make_assistant_line(input_tokens=1, output_tokens=2, cache_creation=3, cache_read=4),
    ])
    os.utime(jsonl_file, (datetime.now().timestamp(), datetime.now().timestamp()))

    stats = manager._extract_session_stats(prompt)

    assert stats is not None
    assert stats.total_input_tokens == 8


def test_extract_stats_last_executed_none(manager, tmp_path):  # SS-019
    """When last_executed is None, returns None immediately."""
    prompt = _make_stats_prompt(tmp_path)
    prompt.last_executed = None

    stats = manager._extract_session_stats(prompt)
    assert stats is None


def test_extract_stats_newest_file_selected(manager, tmp_path, mocker):  # SS-020
    """When multiple JSONL files match, the newest one is used."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    prompt = _make_stats_prompt(tmp_path)
    resolved = prompt._resolved_working_directory
    encoded = resolved.replace("/", "-")
    jsonl_dir = tmp_path / ".claude" / "projects" / encoded
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    older = jsonl_dir / "old-session.jsonl"
    _write_jsonl(older, [
        _make_assistant_line(input_tokens=999, output_tokens=999, cache_creation=0, cache_read=0),
    ])
    old_time = datetime.now().timestamp() - 2
    os.utime(older, (old_time, old_time))

    newer = jsonl_dir / "new-session.jsonl"
    _write_jsonl(newer, [
        _make_assistant_line(input_tokens=1, output_tokens=2, cache_creation=3, cache_read=4),
    ])
    new_time = datetime.now().timestamp()
    os.utime(newer, (new_time, new_time))

    stats = manager._extract_session_stats(prompt)

    assert stats is not None
    assert stats.input_tokens == 1
    assert stats.output_tokens == 2


def test_extract_stats_exception_returns_none(manager, tmp_path):  # SS-021
    """Internal errors are caught and None is returned."""
    prompt = _make_stats_prompt(tmp_path)
    with patch.object(manager, "_do_extract_session_stats", side_effect=OSError("boom")):
        stats = manager._extract_session_stats(prompt)
    assert stats is None


# ===========================================================================
# _format_stats_line()
# ===========================================================================


def test_format_stats_line_with_stats(manager):  # SS-030
    stats = SessionStats(
        input_tokens=100,
        output_tokens=500,
        cache_creation_input_tokens=10000,
        cache_read_input_tokens=5000,
        api_turns=3,
    )
    line = manager._format_stats_line(154.0, stats)
    assert "Duration: 2m" in line
    assert "Input: 15,100 tokens" in line
    assert "Output: 500 tokens" in line
    assert line.startswith("    ")


def test_format_stats_line_without_stats(manager):  # SS-031
    line = manager._format_stats_line(45.0, None)
    assert "Duration: 45s" in line
    assert "Input" not in line
    assert "Output" not in line
    assert line.startswith("    ")


def test_format_stats_line_pipe_separators(manager):  # SS-032
    stats = SessionStats(input_tokens=1, output_tokens=2)
    line = manager._format_stats_line(10.0, stats)
    assert " | " in line


# ===========================================================================
# _log_session_stats()
# ===========================================================================


def test_log_session_stats_detailed_breakdown(manager):  # SS-050
    prompt = QueuedPrompt(id="abc12345", content="test")
    stats = SessionStats(
        input_tokens=402,
        output_tokens=51568,
        cache_creation_input_tokens=19093602,
        cache_read_input_tokens=4255901,
        api_turns=297,
    )

    manager._log_session_stats(prompt, stats)

    assert "402 input" in prompt.execution_log
    assert "19,093,602 cache-write" in prompt.execution_log
    assert "4,255,901 cache-read" in prompt.execution_log
    assert "23,349,905 total input" in prompt.execution_log
    assert "51,568 output" in prompt.execution_log
    assert "297 API turns" in prompt.execution_log


def test_log_session_stats_none_no_log(manager):  # SS-051
    prompt = QueuedPrompt(id="abc12345", content="test")
    manager._log_session_stats(prompt, None)
    assert "Token usage" not in prompt.execution_log


def test_log_session_stats_single_turn_singular(manager):  # SS-052
    prompt = QueuedPrompt(id="abc12345", content="test")
    stats = SessionStats(input_tokens=1, output_tokens=2, api_turns=1)
    manager._log_session_stats(prompt, stats)
    assert "1 API turn)" in prompt.execution_log


# ===========================================================================
# Integration: stats printed in _process_execution_result()
# ===========================================================================


def test_result_success_prints_stats(manager, tmp_path, mocker, capsys):  # SS-040
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_assistant_line(input_tokens=5, output_tokens=50, cache_creation=1000, cache_read=2000),
    ])
    result = ExecutionResult(success=True, output="done", execution_time=120.5)

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "completed successfully" in captured
    assert "Duration:" in captured
    assert "Input: 3,005 tokens" in captured
    assert "Output: 50 tokens" in captured


def test_result_success_no_jsonl_prints_duration_only(manager, tmp_path, mocker, capsys):  # SS-041
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    result = ExecutionResult(success=True, output="done", execution_time=30.0)

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "Duration: 30s" in captured
    assert "Input" not in captured


def test_result_rate_limited_prints_stats_before_cleanup(manager, tmp_path, mocker, capsys):  # SS-042
    """Stats must be extracted BEFORE cleanup deletes the JSONL."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_assistant_line(input_tokens=3, output_tokens=10, cache_creation=500, cache_read=600),
    ])

    rate_info = RateLimitInfo(
        is_rate_limited=True,
        limit_message="usage limit reached",
    )
    result = ExecutionResult(
        success=False,
        output="",
        error="rate limited",
        rate_limit_info=rate_info,
        execution_time=5.0,
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "rate limited" in captured
    assert "Input: 1,103 tokens" in captured
    assert "Output: 10 tokens" in captured


def test_result_generic_failure_retry_prints_stats(manager, tmp_path, mocker, capsys):  # SS-043
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_assistant_line(input_tokens=2, output_tokens=30, cache_creation=100, cache_read=200),
    ])
    result = ExecutionResult(
        success=False, output="", error="something broke", execution_time=10.0
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "failed" in captured
    assert "Input: 302 tokens" in captured
    assert "Output: 30 tokens" in captured


def test_result_generic_failure_permanent_prints_stats(manager, tmp_path, mocker, capsys):  # SS-044
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    prompt.max_retries = 1
    prompt.retry_count = 1
    manager.state.add_prompt(prompt)
    _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_assistant_line(input_tokens=1, output_tokens=5, cache_creation=50, cache_read=100),
    ])
    result = ExecutionResult(
        success=False, output="", error="something broke", execution_time=8.0
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "failed permanently" in captured
    assert "Input: 151 tokens" in captured
    assert "Output: 5 tokens" in captured


def test_result_non_retryable_no_stats_printed(manager, tmp_path, mocker, capsys):  # SS-045
    """Non-retryable errors should not print stats."""
    mocker.patch("claude_code_queue.queue_manager.Path.home", return_value=tmp_path)
    manager.state = manager.storage.load_queue_state()
    prompt = _make_stats_prompt(tmp_path)
    prompt.status = PromptStatus.EXECUTING
    manager.state.add_prompt(prompt)
    _setup_jsonl_for_prompt(tmp_path, prompt, [
        _make_assistant_line(input_tokens=1, output_tokens=1, cache_creation=1, cache_read=1),
    ])
    result = ExecutionResult(
        success=False,
        output="",
        error="nested session",
        execution_time=1.0,
        is_non_retryable=True,
    )

    manager._process_execution_result(prompt, result)

    captured = capsys.readouterr().out
    assert "non-retryable" in captured
    assert "Input" not in captured
    assert "Duration" not in captured

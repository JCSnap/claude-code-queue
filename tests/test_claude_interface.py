"""
Tests for claude_interface.py — ClaudeCodeInterface (subprocess mocked).

All tests use the `interface` fixture which patches _verify_claude_available
so the constructor never actually calls the claude binary.
"""

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_code_queue.claude_interface import (
    ClaudeCodeInterface,
    _RATE_LIMIT_MAX_RESET_HOURS,
    _RATE_LIMIT_SCAN_CHARS,
)
from claude_code_queue.models import QueuedPrompt, RateLimitInfo, PromptStatus


# ===========================================================================
# Rate-Limit Detection
# ===========================================================================


def test_rate_limit_detected_usage_limit_reached(interface):  # CLI-001
    """'usage limit reached' pattern triggers is_rate_limited=True."""
    result = interface._detect_rate_limit("usage limit reached|1735689600")
    assert result.is_rate_limited is True


def test_rate_limit_detected_rate_limit_exceeded(interface):  # CLI-002
    """'rate limit exceeded' pattern triggers is_rate_limited=True."""
    result = interface._detect_rate_limit(
        "Error: rate limit exceeded, please try again"
    )
    assert result.is_rate_limited is True


def test_rate_limit_detected_too_many_requests(interface):  # CLI-003
    """'429 too many requests' pattern triggers is_rate_limited=True."""
    result = interface._detect_rate_limit("429 Too Many Requests")
    assert result.is_rate_limited is True


def test_rate_limit_not_detected_normal_output(interface):  # CLI-004
    """Normal Claude output does not trigger rate-limit detection."""
    result = interface._detect_rate_limit(
        "Successfully refactored the authentication module."
    )
    assert result.is_rate_limited is False


def test_rate_limit_detection_is_case_insensitive(interface):  # CLI-005
    """Detection uses output.lower() — mixed-case strings are caught."""
    result = interface._detect_rate_limit("Usage Limit Reached — please wait")
    assert result.is_rate_limited is True

    result2 = interface._detect_rate_limit("RATE LIMIT EXCEEDED")
    assert result2.is_rate_limited is True


def test_rate_limit_detected_quota_exceeded(interface):  # CLI-006
    """'quota exceeded' pattern triggers is_rate_limited=True."""
    result = interface._detect_rate_limit("quota exceeded for this billing period")
    assert result.is_rate_limited is True


def test_limit_exceeded_without_rate_qualifier_not_detected(interface):  # S11a
    """Bare 'limit exceeded' without a rate/quota qualifier must NOT trigger
    detection. The pattern was removed in S11a to prevent false positives from
    system errors and tool output.
    """
    false_positive_strings = [
        "API limit exceeded, try again later",
        "Error: maximum recursion depth exceeded",
        "MemoryError: memory limit exceeded",
        "OSError: file size limit exceeded",
        "stack limit exceeded during compilation",
    ]
    for text in false_positive_strings:
        result = interface._detect_rate_limit(text)
        assert result.is_rate_limited is False, (
            f"False positive triggered by: {text!r}"
        )


def test_rate_limit_sets_limit_message_and_truncates(interface):  # CLI-008
    """_detect_rate_limit() captures output as limit_message, truncated to 500 chars."""
    long_output = "usage limit reached. " + "x" * 600
    result = interface._detect_rate_limit(long_output)
    assert result.is_rate_limited is True
    assert result.limit_message != ""
    assert len(result.limit_message) <= 500
    assert "usage limit reached" in result.limit_message.lower()


def test_rate_limit_detected_in_stderr(interface):  # CLI-009
    """Rate-limit message in stderr is detected.

    execute_prompt() passes result.stderr (only) to _detect_rate_limit(),
    so messages in stderr are correctly caught.
    """
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="usage limit reached"
        )
        result = interface.execute_prompt(QueuedPrompt(content="task"))
    assert result.rate_limit_info.is_rate_limited is True
    assert result.success is False


def test_rate_limit_detected_in_stderr_only(mocker):  # CLI-010
    """Rate-limit phrase in stderr → is_rate_limited=True."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="usage limit reached"
        )
        result = iface.execute_prompt(QueuedPrompt(content="task"))

    assert result.rate_limit_info.is_rate_limited is True


def test_rate_limit_in_stdout_only_not_detected(mocker):  # CLI-011
    """Rate-limit phrase only in stdout → is_rate_limited=False (no false positive)."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="usage limit reached", stderr=""
        )
        result = iface.execute_prompt(QueuedPrompt(content="task"))

    assert result.rate_limit_info.is_rate_limited is False
    assert result.success is True


# ===========================================================================
# Reset Time Parsing
# ===========================================================================


def test_extract_reset_time_parses_unix_pipe_format(interface):  # CLI-012
    """'usage limit reached|<unix_ts>' is parsed to the correct datetime."""
    ts = int(datetime(2025, 6, 1, 15, 0, 0).timestamp())
    result = interface._extract_reset_time_from_limit_message(
        f"usage limit reached|{ts}"
    )
    assert result is not None
    delta = abs((result - datetime(2025, 6, 1, 15, 0, 0)).total_seconds())
    assert delta < 2, f"Parsed time differs by {delta:.1f}s from expected"


def test_extract_reset_time_falls_back_to_estimate(interface):  # CLI-013
    """When no timestamp is found, the method falls back to _estimate_reset_time()
    which always returns a future datetime.
    """
    result = interface._extract_reset_time_from_limit_message(
        "usage limit reached (no timestamp here)"
    )
    assert result is not None
    assert result > datetime.now(), (
        "Fallback estimate must be a future datetime"
    )


def test_extract_reset_time_parses_iso_datetime(interface):  # CLI-014
    """ISO datetime with Z suffix is parsed; the result is a naive datetime."""
    iso_output = "usage limit reached. Resets at 2025-06-01T10:00:00Z."
    result = interface._extract_reset_time_from_limit_message(iso_output)
    assert result is not None
    assert result.year == 2025
    assert result.month == 6
    assert result.tzinfo is None, "Result must be a naive datetime (no tzinfo)"


# ===========================================================================
# Reset Time Estimation
# ===========================================================================
#
# _estimate_reset_time() never calls datetime() as a constructor — it only
# calls datetime.now() and then uses .replace() / arithmetic on the result.
# timedelta is imported separately and is NOT affected by the patch.
#
# Pattern for each test:
#   with patch('claude_code_queue.claude_interface.datetime') as mock_dt:
#       mock_dt.now.return_value = datetime(2025, 1, 1, HOUR, MINUTE, SECOND)
#       result = interface._estimate_reset_time("")
#   # assert OUTSIDE the with-block so the mock does not intercept datetime(...)
#   assert result == datetime(2025, 1, EXPECTED_DAY, EXPECTED_HOUR, 0, 0)


def test_estimate_reset_time_hour_0(interface):  # CLI-015
    """00:30  →  05:00 today (first reset window)."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 0, 30, 0)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 1, 5, 0, 0)


def test_estimate_reset_time_hour_4(interface):  # CLI-016
    """04:59  →  05:00 today (still before first window close)."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 4, 59, 0)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 1, 5, 0, 0)


def test_estimate_reset_time_hour_5(interface):  # CLI-017
    """05:00  →  10:00 today (just entered the second window)."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 5, 0, 0)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 1, 10, 0, 0)


def test_estimate_reset_time_hour_9(interface):  # CLI-018
    """09:59  →  10:00 today (still in second window)."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 9, 59, 0)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 1, 10, 0, 0)


def test_estimate_reset_time_hour_10(interface):  # CLI-019
    """10:00  →  15:00 today (just entered the third window)."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 10, 0, 0)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 1, 15, 0, 0)


def test_estimate_reset_time_hour_15(interface):  # CLI-020
    """15:00  →  20:00 today (just entered the fourth window)."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 15, 0, 0)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 1, 20, 0, 0)


def test_estimate_reset_time_hour_20(interface):  # CLI-021
    """20:00  →  01:00 tomorrow."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 20, 0, 0)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 2, 1, 0, 0)


def test_estimate_reset_time_hour_23(interface):  # CLI-022
    """23:59  →  01:00 tomorrow (same late-night window as 20:00)."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 23, 59, 0)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 2, 1, 0, 0)


def test_estimate_reset_time_at_exact_boundary_hour_5(interface):  # CLI-023
    """Exactly 05:00:00 (the >= 5 boundary) → 10:00 today."""
    with patch("claude_code_queue.claude_interface.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2025, 1, 1, 5, 0, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = interface._estimate_reset_time("")
    assert result == datetime(2025, 1, 1, 10, 0, 0)


# ===========================================================================
# Command Execution
# ===========================================================================


def test_execute_prompt_calls_claude_with_print_flag(interface):  # CLI-024
    """execute_prompt() calls the claude binary with --print."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        interface.execute_prompt(QueuedPrompt(content="test task"))
        args = mock_run.call_args[0][0]
        assert "--print" in args


def test_execute_prompt_includes_dangerously_skip_permissions(interface):  # CLI-025
    """execute_prompt() includes --dangerously-skip-permissions in the command."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        interface.execute_prompt(QueuedPrompt(content="test task"))
        args = mock_run.call_args[0][0]
        assert "--dangerously-skip-permissions" in args


def test_execute_prompt_success_returns_success_result(interface):  # CLI-026
    """returncode=0 with no rate-limit output → success=True."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="All done", stderr="")
        result = interface.execute_prompt(QueuedPrompt(content="task"))
        assert result.success is True
        assert result.output == "All done"


def test_execute_prompt_failure_returns_failure_result(interface):  # CLI-027
    """Non-zero returncode → success=False."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Something went wrong"
        )
        result = interface.execute_prompt(QueuedPrompt(content="task"))
        assert result.success is False


def test_execute_prompt_rate_limit_in_stdout_not_detected(interface):  # CLI-028
    """Rate-limit text only in stdout is NOT detected (stderr-only detection)."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="usage limit reached", stderr=""
        )
        result = interface.execute_prompt(QueuedPrompt(content="task"))
        assert result.rate_limit_info.is_rate_limited is False
        assert result.success is True


def test_execute_prompt_with_context_files_includes_at_references(
    interface, tmp_path
):  # CLI-029
    """Existing context_files entries are passed as '@filename' references."""
    context_file = tmp_path / "README.md"
    context_file.write_text("# README")

    prompt = QueuedPrompt(
        content="task",
        working_directory=str(tmp_path),
        context_files=["README.md"],
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        interface.execute_prompt(prompt)
        call_args = mock_run.call_args[0][0]
        full_cmd = " ".join(call_args)
        assert "@README.md" in full_cmd, (
            f"Expected '@README.md' in command args: {call_args}"
        )


def test_execute_prompt_uses_working_directory(interface, tmp_path):  # CLI-030
    """execute_prompt() passes cwd= to subprocess.run instead of os.chdir()."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        prompt = QueuedPrompt(content="task", working_directory=str(tmp_path))
        interface.execute_prompt(prompt)

    expected = str(tmp_path.resolve())
    call_kwargs = mock_run.call_args[1]
    assert "cwd" in call_kwargs, (
        f"Expected 'cwd' kwarg in subprocess.run call; got kwargs: {call_kwargs}"
    )
    assert call_kwargs["cwd"] == expected, (
        f"Expected cwd={expected!r}; got cwd={call_kwargs['cwd']!r}"
    )


def test_execute_prompt_skips_nonexistent_context_files(interface, tmp_path):  # CLI-031
    """Context file paths that don't exist on disk are omitted from the command."""
    prompt = QueuedPrompt(
        content="task",
        working_directory=str(tmp_path),
        context_files=["nonexistent.py", "also-missing.py"],
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        interface.execute_prompt(prompt)
        call_args = mock_run.call_args[0][0]
        full_cmd = " ".join(call_args)
        assert "@nonexistent.py" not in full_cmd
        assert "@also-missing.py" not in full_cmd


def test_execute_prompt_timeout_returns_failure_result(interface):  # CLI-032
    """subprocess.TimeoutExpired → success=False with 'timed out' in error."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)
        result = interface.execute_prompt(QueuedPrompt(content="task"))
        assert result.success is False
        assert "timed out" in result.error.lower(), (
            f"Expected 'timed out' in error message, got: {result.error!r}"
        )


# ===========================================================================
# Connection Testing
# ===========================================================================


def test_test_connection_returns_true_when_claude_available(interface):  # CLI-033
    """test_connection() returns (True, msg) when the subprocess exits 0."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Claude Code v1.0", stderr=""
        )
        ok, msg = interface.test_connection()
        assert ok is True


def test_test_connection_returns_false_when_unavailable(interface):  # CLI-034
    """test_connection() returns (False, msg) with 'not found' when FileNotFoundError."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("claude not found")
        ok, msg = interface.test_connection()
        assert ok is False
        assert "not found" in msg.lower(), (
            f"Expected 'not found' in error message, got: {msg!r}"
        )


def test_version_warning_emitted_for_old_claude(mocker, capsys):  # CLI-035
    """A version string older than (2,1,50) triggers a warning on stderr."""
    mocker.patch(
        "subprocess.run",
        return_value=MagicMock(
            returncode=0,
            stdout="1.0.0 (Claude Code)",
            stderr="",
        ),
    )
    ClaudeCodeInterface(claude_command="claude", timeout=60)
    captured = capsys.readouterr()
    assert "Warning" in captured.err
    assert "2.1.50" in captured.err


def test_version_warning_not_emitted_for_current_claude(mocker, capsys):  # CLI-036
    """A version string >= (2,1,50) does NOT trigger a warning."""
    mocker.patch(
        "subprocess.run",
        return_value=MagicMock(
            returncode=0,
            stdout="2.1.50 (Claude Code)",
            stderr="",
        ),
    )
    ClaudeCodeInterface(claude_command="claude", timeout=60)
    captured = capsys.readouterr()
    assert "Warning" not in captured.err


# ===========================================================================
# Security & Configuration
# ===========================================================================


def test_skip_permissions_true_includes_flag(mocker):  # CLI-037
    """skip_permissions=True (default) → --dangerously-skip-permissions in cmd."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60, skip_permissions=True)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        iface.execute_prompt(QueuedPrompt(content="task"))
        args = mock_run.call_args[0][0]

    assert "--dangerously-skip-permissions" in args


def test_skip_permissions_false_omits_flag(mocker):  # CLI-038
    """skip_permissions=False → --dangerously-skip-permissions NOT in cmd."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60, skip_permissions=False)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
        iface.execute_prompt(QueuedPrompt(content="task"))
        args = mock_run.call_args[0][0]

    assert "--dangerously-skip-permissions" not in args
    assert "--print" in args


def test_out_of_home_working_directory_emits_warning(mocker, tmp_path, capsys):  # CLI-039
    """working_directory outside home → warning on stderr."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60)

    prompt = QueuedPrompt(content="task", working_directory="/tmp")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch("pathlib.Path.home", return_value=Path("/home/testuser")):
            iface.execute_prompt(prompt)

    captured = capsys.readouterr()
    assert "Warning" in captured.err or "warning" in captured.err.lower(), (
        f"Expected a warning for out-of-home path, got stderr: {captured.err!r}"
    )


def test_in_home_working_directory_no_warning(mocker, tmp_path, capsys):  # CLI-040
    """working_directory inside home → no warning on stderr."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60)

    prompt = QueuedPrompt(content="task", working_directory=str(tmp_path))

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with patch("pathlib.Path.home", return_value=tmp_path.parent):
            iface.execute_prompt(prompt)

    captured = capsys.readouterr()
    assert "Warning" not in captured.err, (
        f"Expected no warning for in-home path, got stderr: {captured.err!r}"
    )


def test_cap_reset_time_limits_far_future_timestamp(mocker):  # CLI-041
    """reset_time far in the future is capped to <= 24 hours from now."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")

    far_future = datetime.now() + timedelta(days=7)
    capped = ClaudeCodeInterface._cap_reset_time(far_future)

    max_allowed = datetime.now() + timedelta(hours=_RATE_LIMIT_MAX_RESET_HOURS)
    assert capped <= max_allowed, (
        f"Capped time {capped} exceeds allowed maximum {max_allowed}"
    )
    delta = (max_allowed - capped).total_seconds()
    assert delta < 5, f"Capped time {capped} is not near the 24h cap"


def test_cap_reset_time_strips_timezone_info(mocker):  # CLI-042
    """_cap_reset_time() returns a naive datetime regardless of input tzinfo."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")

    aware_dt = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    result = ClaudeCodeInterface._cap_reset_time(aware_dt)

    assert result.tzinfo is None, "Result must be a naive datetime (no tzinfo)"


def test_extract_reset_time_caps_far_future_unix_timestamp(mocker):  # CLI-043
    """A unix timestamp 7 days away is capped to <= 24h by _extract_reset_time."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60)

    far_ts = int((datetime.now() + timedelta(days=7)).timestamp())
    result = iface._extract_reset_time_from_limit_message(
        f"usage limit reached|{far_ts}"
    )

    assert result is not None
    max_allowed = datetime.now() + timedelta(hours=_RATE_LIMIT_MAX_RESET_HOURS)
    assert result <= max_allowed, (
        f"Reset time {result} not capped to <= {max_allowed}"
    )


def test_estimate_reset_time_caps_at_24h(mocker):  # CLI-044
    """_estimate_reset_time() result is always <= 24 hours from now."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60)

    result = iface._estimate_reset_time("")

    max_allowed = datetime.now() + timedelta(hours=_RATE_LIMIT_MAX_RESET_HOURS)
    assert result <= max_allowed + timedelta(seconds=1), (
        f"Estimated reset {result} exceeds 24h cap {max_allowed}"
    )


# ===========================================================================
# Scan-Window Depth Guard (S11b)
# ===========================================================================


def test_detect_rate_limit_ignores_phrase_beyond_scan_window(interface):  # S11b
    """A rate-limit phrase that appears only AFTER _RATE_LIMIT_SCAN_CHARS characters
    of stderr must NOT trigger detection.

    Scenario: a long-running task invokes a subprocess whose verbose debug log
    happens to contain 'quota exceeded' deep in its output. The depth cap
    ensures this does not stall the queue.
    """
    prefix = "x" * _RATE_LIMIT_SCAN_CHARS  # pushes phrase to index _RATE_LIMIT_SCAN_CHARS
    result = interface._detect_rate_limit(prefix + "\nquota exceeded")
    assert result.is_rate_limited is False, (
        "Pattern beyond scan window must not trigger rate-limit detection"
    )


def test_detect_rate_limit_still_fires_within_scan_window(interface):  # S11b
    """A rate-limit phrase that appears WITHIN _RATE_LIMIT_SCAN_CHARS characters
    is still detected correctly after the scan-window restriction.
    """
    result = interface._detect_rate_limit("quota exceeded for this billing period")
    assert result.is_rate_limited is True


def test_detect_rate_limit_fires_at_last_char_of_scan_window(interface):  # S11b
    """A pattern whose final character sits at index _RATE_LIMIT_SCAN_CHARS - 1
    (the last position inside the window) is still detected.

    Validates the boundary is inclusive: output[:N] includes index N-1.
    """
    pattern = "quota exceeded"
    # Pad prefix so pattern ends exactly at index _RATE_LIMIT_SCAN_CHARS - 1.
    prefix = "x" * (_RATE_LIMIT_SCAN_CHARS - len(pattern))
    result = interface._detect_rate_limit(prefix + pattern + "x" * 1000)
    assert result.is_rate_limited is True, (
        "Pattern ending at the last character of the scan window must still fire"
    )


# ===========================================================================
# False-Positive Regression Suite (S11c)
# ===========================================================================


@pytest.mark.parametrize("fp_text", [
    # ------------------------------------------------------------------
    # Group 1: no matching pattern — safe regardless of scan-window size
    # ------------------------------------------------------------------
    # Python runtime errors
    "Traceback (most recent call last):\n  ...\nRecursionError: maximum recursion depth exceeded",
    # Shell / OS errors
    "bash: fork: retry: Resource temporarily unavailable",
    "ulimit: file size: cannot modify limit: Operation not permitted",
    # Compiler / linker output (contains "limit exceeded" — safe after S11a removes it)
    "ld: warning: stack size limit exceeded, consider reducing stack usage",
    # ------------------------------------------------------------------
    # Group 2: live pattern buried beyond _RATE_LIMIT_SCAN_CHARS — tests S11b
    # Short prose versions of these strings ARE accepted false positives for
    # this pass (see "Known Remaining Limitation" in checklist).
    # ------------------------------------------------------------------
    pytest.param(
        "x" * _RATE_LIMIT_SCAN_CHARS + "\nquota exceeded for external service",
        id="quota-exceeded-beyond-scan-window",
    ),
    pytest.param(
        "x" * _RATE_LIMIT_SCAN_CHARS + "\ntoo many requests to upstream service",
        id="too-many-requests-beyond-scan-window",
    ),
])
def test_false_positive_not_detected(interface, fp_text):  # S11c
    """Common false-positive strings must NOT trigger rate-limit detection.

    After R3, detection runs on stderr only; after S11a, 'limit exceeded' is
    removed; after S11b, only the first _RATE_LIMIT_SCAN_CHARS characters are
    scanned. This parametrized test guards against regression of any of those
    hardening measures.
    """
    result = interface._detect_rate_limit(fp_text)
    assert result.is_rate_limited is False, (
        f"False positive triggered by:\n  {fp_text!r}"
    )

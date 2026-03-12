"""
Tests for queue_manager.py — QueueManager (ClaudeCodeInterface mocked).

ClaudeCodeInterface.execute_prompt is patched for all execution tests.
The manager fixture uses tmp_path so every test gets a fresh storage dir.
"""

import signal
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from claude_code_queue.claude_interface import ClaudeCodeInterface
from claude_code_queue.models import (
    ExecutionResult,
    PromptStatus,
    QueuedPrompt,
    QueueState,
    RateLimitInfo,
)
from claude_code_queue.queue_manager import QueueManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _success_result(output: str = "done") -> ExecutionResult:
    return ExecutionResult(success=True, output=output, error="", execution_time=0.1)


def _fail_result(error: str = "error") -> ExecutionResult:
    return ExecutionResult(success=False, output="", error=error, execution_time=0.1)


def _rate_limit_result(reset_time=None) -> ExecutionResult:
    return ExecutionResult(
        success=False,
        output="usage limit reached",
        error="",
        rate_limit_info=RateLimitInfo(is_rate_limited=True, reset_time=reset_time),
        execution_time=0.1,
    )


def _non_retryable_result(error: str = "Error: Claude Code cannot be launched inside another Claude Code session.") -> ExecutionResult:
    return ExecutionResult(
        success=False,
        output="",
        error=error,
        execution_time=0.4,
        is_non_retryable=True,
    )


# ===========================================================================
# Rate-Limit Result Processing
# ===========================================================================


def test_was_already_rate_limited_uses_rate_limited_at(manager):  # QMG-001
    """_process_execution_result() must use prompt.rate_limited_at (not
    prompt.status) to decide whether to increment rate_limited_count.
    """
    prompt = QueuedPrompt(content="task", status=PromptStatus.EXECUTING)
    prompt.rate_limited_at = datetime.now() - timedelta(minutes=10)

    manager.state = QueueState()
    manager.state.add_prompt(prompt)
    manager.state.rate_limited_count = 1

    rl_result = _rate_limit_result()
    manager._process_execution_result(prompt, rl_result)

    assert manager.state.rate_limited_count == 1, (
        "rate_limited_count must NOT be incremented when the prompt was already "
        "rate-limited (rate_limited_at is set)"
    )


def test_reset_time_assigned_from_rate_limit_info(manager):  # QMG-002
    """prompt.reset_time must be set from rate_limit_info.reset_time."""
    expected_reset = datetime(2025, 6, 1, 15, 0, 0)
    prompt = QueuedPrompt(content="task", status=PromptStatus.EXECUTING)
    manager.state = QueueState()
    manager.state.add_prompt(prompt)

    rl_result = _rate_limit_result(reset_time=expected_reset)
    manager._process_execution_result(prompt, rl_result)

    assert prompt.reset_time is not None, "reset_time must be set on the prompt"
    delta = abs((prompt.reset_time - expected_reset).total_seconds())
    assert delta < 1, f"reset_time drifted by {delta:.3f}s"
    assert prompt.reset_time.tzinfo is None, "reset_time must be stored as naive"


def test_check_rate_limited_uses_reset_time_when_past(manager):  # QMG-003
    """When reset_time is in the past, re-queue immediately — even if
    rate_limited_at was only 30 seconds ago (inside the 5-min heuristic window).
    """
    prompt = QueuedPrompt(content="task", status=PromptStatus.RATE_LIMITED, max_retries=3)
    prompt.rate_limited_at = datetime.now() - timedelta(seconds=30)
    prompt.reset_time = datetime.now() - timedelta(minutes=1)
    prompt.retry_count = 0

    manager.state = QueueState()
    manager.state.add_prompt(prompt)
    manager._check_rate_limited_prompts()

    assert prompt.status == PromptStatus.QUEUED, (
        "Prompt with a past reset_time must be re-queued regardless of "
        "how recently it was rate-limited"
    )


def test_check_rate_limited_stays_rate_limited_when_reset_time_future(manager):  # QMG-004
    """When reset_time is known but future, the prompt must NOT be re-queued
    by the 5-minute heuristic fallback.
    """
    prompt = QueuedPrompt(content="task", status=PromptStatus.RATE_LIMITED, max_retries=3)
    prompt.rate_limited_at = datetime.now() - timedelta(minutes=6)
    prompt.reset_time = datetime.now() + timedelta(hours=2)
    prompt.retry_count = 0

    manager.state = QueueState()
    manager.state.add_prompt(prompt)
    manager._check_rate_limited_prompts()

    assert prompt.status == PromptStatus.RATE_LIMITED, (
        "Prompt with a future reset_time must stay RATE_LIMITED "
        "(the 5-min heuristic must not fire when reset_time is known)"
    )


def test_check_rate_limited_falls_back_to_5min_heuristic_when_no_reset_time(manager):  # QMG-005
    """When no reset_time is set and rate_limited_at > 5 min ago, re-queue."""
    prompt = QueuedPrompt(content="task", status=PromptStatus.RATE_LIMITED, max_retries=3)
    prompt.rate_limited_at = datetime.now() - timedelta(minutes=6)
    prompt.reset_time = None
    prompt.retry_count = 0

    manager.state = QueueState()
    manager.state.add_prompt(prompt)
    manager._check_rate_limited_prompts()

    assert prompt.status == PromptStatus.QUEUED


def test_state_saved_after_rate_limit_check_exhausts_retries(manager):  # QMG-006
    """save_queue_state() is called even when no prompt is executed.

    Scenario: a rate-limited prompt exhausts its retries during
    _check_rate_limited_prompts() → becomes FAILED → state must be persisted.
    """
    state = manager.storage.load_queue_state()
    prompt = QueuedPrompt(
        content="task", status=PromptStatus.RATE_LIMITED, max_retries=1
    )
    prompt.rate_limited_at = datetime.now() - timedelta(minutes=10)
    prompt.retry_count = 1
    state.add_prompt(prompt)
    manager.storage.save_queue_state(state)

    with patch.object(
        manager.storage, "save_queue_state", wraps=manager.storage.save_queue_state
    ) as mock_save:
        manager.state = None
        manager._process_queue_iteration()
        mock_save.assert_called()

    failed_files = list(manager.storage.failed_dir.glob("*.md"))
    assert len(failed_files) == 1, (
        f"Expected 1 failed file, found {len(failed_files)}: {failed_files}"
    )


def test_check_rate_limited_fails_prompt_when_retries_exhausted(manager):  # QMG-007
    """When retries are exhausted during the 5-min check, status → FAILED."""
    prompt = QueuedPrompt(
        content="task", status=PromptStatus.RATE_LIMITED, max_retries=1
    )
    prompt.rate_limited_at = datetime.now() - timedelta(minutes=10)
    prompt.reset_time = None
    prompt.retry_count = 1

    manager.state = QueueState()
    manager.state.add_prompt(prompt)
    manager._check_rate_limited_prompts()

    assert prompt.status == PromptStatus.FAILED, (
        f"Expected FAILED when retries exhausted, got {prompt.status}"
    )


# ===========================================================================
# Execution Lifecycle
# ===========================================================================


def test_start_processes_queued_prompt(manager):  # QMG-008
    """A single QUEUED prompt with a mocked successful result moves to COMPLETED."""
    success = _success_result("All done")
    with patch.object(manager.claude_interface, "execute_prompt", return_value=success):
        state = manager.storage.load_queue_state()
        state.add_prompt(QueuedPrompt(content="test task"))
        manager.storage.save_queue_state(state)
        manager._process_queue_iteration()
        completed = list(manager.storage.completed_dir.glob("*.md"))
        assert len(completed) == 1


def test_priority_order_respected(manager):  # QMG-009
    """The prompt with the lowest priority number is executed first."""
    executed_ids = []

    def fake_execute(prompt):
        executed_ids.append(prompt.id)
        return _success_result()

    state = manager.storage.load_queue_state()
    p_low = QueuedPrompt(content="low priority", priority=5)
    p_high = QueuedPrompt(content="high priority", priority=1)
    state.add_prompt(p_low)
    state.add_prompt(p_high)
    manager.storage.save_queue_state(state)
    manager.state = None

    with patch.object(manager.claude_interface, "execute_prompt", side_effect=fake_execute):
        manager._process_queue_iteration()

    assert len(executed_ids) >= 1, "At least one prompt must have been executed"
    assert executed_ids[0] == p_high.id, (
        f"Expected high-priority prompt ({p_high.id}) first, got {executed_ids[0]}"
    )


def test_failed_prompt_retried_up_to_max_retries(manager):  # QMG-010
    """A prompt that always fails is retried up to max_retries times then FAILED.

    After Fix 3, each failure sets retry_not_before into the future. We clear
    it between iterations to simulate the passage of time.
    """
    fail = _fail_result("build failed")
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="failing task", max_retries=2)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)

    with patch.object(manager.claude_interface, "execute_prompt", return_value=fail):
        for _ in range(5):
            manager.state = None
            manager._process_queue_iteration()
            # Simulate passage of time past retry_not_before.
            if manager.state:
                for pr in manager.state.prompts:
                    if pr.retry_not_before is not None:
                        pr.retry_not_before = datetime.now() - timedelta(seconds=1)
                manager.storage.save_queue_state(manager.state)

    failed_files = list(manager.storage.failed_dir.glob("*.md"))
    assert len(failed_files) == 1, (
        f"Expected 1 failed file after exhausting retries, got {len(failed_files)}"
    )


def test_rate_limited_prompt_re_queued_after_reset_time(manager):  # QMG-011
    """A rate-limited prompt with a past reset_time is re-queued and then completes."""
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="rate limited task", max_retries=3)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)

    past_reset = datetime.now() - timedelta(minutes=1)
    rl_result = _rate_limit_result(reset_time=past_reset)

    with patch.object(manager.claude_interface, "execute_prompt", return_value=rl_result):
        manager.state = None
        manager._process_queue_iteration()

    state_after_rl = manager.storage.load_queue_state()
    rl_prompts = [pr for pr in state_after_rl.prompts if pr.status == PromptStatus.RATE_LIMITED]
    assert len(rl_prompts) == 1, "Prompt should be RATE_LIMITED after first iteration"
    assert rl_prompts[0].reset_time is not None, "reset_time must have been assigned"
    assert rl_prompts[0].reset_time <= datetime.now(), "reset_time should be in the past"

    success = _success_result("done")
    with patch.object(manager.claude_interface, "execute_prompt", return_value=success):
        manager.state = None
        manager._process_queue_iteration()

    completed = list(manager.storage.completed_dir.glob("*.md"))
    assert len(completed) == 1, "Prompt should complete after re-queue and successful execution"


def test_cancel_removes_prompt_from_execution(manager):  # QMG-012
    """remove_prompt() cancels a QUEUED prompt; it disappears from the active queue."""
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="to be cancelled")
    state.add_prompt(p)
    manager.storage.save_queue_state(state)

    result = manager.remove_prompt(p.id)
    assert result is True

    reloaded = manager.storage.load_queue_state()
    remaining = [pr for pr in reloaded.prompts if pr.id == p.id]
    assert len(remaining) == 0 or remaining[0].status == PromptStatus.CANCELLED


def test_log_shows_infinity_for_max_retries_minus_one(manager):  # QMG-013
    """When max_retries=-1, the execution log must display '∞' rather than '-1'."""
    fail = _fail_result("err")
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="unlimited retries", max_retries=-1)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)
    manager.state = None

    with patch.object(manager.claude_interface, "execute_prompt", return_value=fail):
        manager._process_queue_iteration()

    executed_prompt = next(
        (pr for pr in manager.state.prompts if pr.id == p.id), None
    )
    assert executed_prompt is not None, "Prompt must be in manager.state after iteration"
    assert "∞" in executed_prompt.execution_log, (
        f"Expected '∞' in execution_log for max_retries=-1, got:\n{executed_prompt.execution_log}"
    )


def test_get_status_returns_correct_counts(manager):  # QMG-014
    """get_status() returns the live QueueState with the correct prompt list."""
    state = manager.storage.load_queue_state()
    state.add_prompt(QueuedPrompt(content="task1"))
    state.add_prompt(QueuedPrompt(content="task2"))
    manager.storage.save_queue_state(state)

    status = manager.get_status()
    queued = [p for p in status.prompts if p.status == PromptStatus.QUEUED]
    assert len(queued) == 2


def test_executing_to_queued_retry_after_failure(manager):  # QMG-015
    """After a failed execution with retries remaining, prompt transitions
    EXECUTING → QUEUED (not EXECUTING → FAILED).
    """
    fail = _fail_result("transient error")
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="retry me", max_retries=3, retry_count=0)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)
    manager.state = None

    with patch.object(manager.claude_interface, "execute_prompt", return_value=fail):
        manager._process_queue_iteration()

    updated = next(
        (pr for pr in manager.state.prompts if pr.id == p.id), None
    )
    assert updated is not None, "Prompt must still be in manager.state"
    assert updated.status == PromptStatus.QUEUED, (
        f"Expected QUEUED after first failure, got {updated.status}"
    )
    assert updated.retry_count == 1, (
        f"Expected retry_count=1 after one failure, got {updated.retry_count}"
    )


def test_timezone_aware_reset_time_stored_as_naive(manager):  # QMG-016
    """Timezone-aware reset_time from rate_limit_info is stripped of tzinfo."""
    aware_time = datetime(2025, 6, 1, 15, 0, 0, tzinfo=timezone.utc)
    rl_result = ExecutionResult(
        success=False,
        output="usage limit reached",
        error="",
        rate_limit_info=RateLimitInfo(is_rate_limited=True, reset_time=aware_time),
        execution_time=0.1,
    )
    prompt = QueuedPrompt(content="task", status=PromptStatus.EXECUTING)
    manager.state = QueueState()
    manager.state.add_prompt(prompt)

    manager._process_execution_result(prompt, rl_result)

    assert prompt.reset_time is not None, "reset_time must be assigned"
    assert prompt.reset_time.tzinfo is None, (
        "reset_time must be stored as naive (no tzinfo)"
    )


def test_cancel_executing_prompt_returns_false(manager):  # QMG-017
    """remove_prompt() refuses to cancel a currently-executing prompt."""
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="running now", status=PromptStatus.EXECUTING)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)
    manager.state = state

    result = manager.remove_prompt(p.id)
    assert result is False


def test_shutdown_requeues_executing_prompts(manager):  # QMG-018
    """_shutdown() transitions EXECUTING prompts back to QUEUED for the next run."""
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="in flight", status=PromptStatus.EXECUTING)
    state.add_prompt(p)
    manager.state = state

    manager._shutdown()

    reloaded = manager.storage.load_queue_state()
    prompt_after = next(
        (pr for pr in reloaded.prompts if pr.id == p.id), None
    )
    assert prompt_after is not None, "Prompt must survive shutdown"
    assert prompt_after.status == PromptStatus.QUEUED, (
        f"Expected QUEUED after shutdown, got {prompt_after.status}"
    )


def test_process_execution_result_assigns_last_processed(manager):  # QMG-019
    """After any execution (success or failure), state.last_processed is updated."""
    manager.state = QueueState()
    prompt = QueuedPrompt(content="task")
    manager.state.add_prompt(prompt)
    assert manager.state.last_processed is None

    success = _success_result("ok")
    manager._process_execution_result(prompt, success)

    assert manager.state.last_processed is not None
    delta = abs((manager.state.last_processed - datetime.now()).total_seconds())
    assert delta < 5, f"last_processed is {delta:.1f}s from now; expected < 5s"


# ===========================================================================
# Prompt Management
# ===========================================================================


def test_remove_nonexistent_prompt_returns_false(manager):  # QMG-020
    """remove_prompt() with an unknown id returns False (not raises)."""
    manager.state = manager.storage.load_queue_state()
    result = manager.remove_prompt("doesnotexist")
    assert result is False


def test_iteration_preserves_in_memory_counters_across_reload(manager):  # QMG-021
    """_process_queue_iteration() must not regress in-memory counters that are
    ahead of the on-disk value when reloading state.
    """
    state = manager.storage.load_queue_state()
    state.total_processed = 3
    manager.storage.save_queue_state(state)

    manager.state = state
    manager.state.total_processed = 7

    manager._process_queue_iteration()

    assert manager.state.total_processed == 7, (
        "In-memory counter must not be overwritten by stale on-disk value"
    )


def test_check_rate_limited_stays_rate_limited_under_5min(manager):  # QMG-022
    """4 minutes 59 seconds is not long enough — prompt stays RATE_LIMITED."""
    prompt = QueuedPrompt(
        content="task", status=PromptStatus.RATE_LIMITED, max_retries=3
    )
    prompt.rate_limited_at = datetime.now() - timedelta(seconds=299)
    prompt.reset_time = None
    prompt.retry_count = 0

    manager.state = QueueState()
    manager.state.add_prompt(prompt)
    manager._check_rate_limited_prompts()

    assert prompt.status == PromptStatus.RATE_LIMITED


def test_check_rate_limited_requeues_just_over_5min(manager):  # QMG-023
    """5 minutes 1 second is enough — prompt re-queues via the heuristic."""
    prompt = QueuedPrompt(
        content="task", status=PromptStatus.RATE_LIMITED, max_retries=3
    )
    prompt.rate_limited_at = datetime.now() - timedelta(seconds=301)
    prompt.reset_time = None
    prompt.retry_count = 0

    manager.state = QueueState()
    manager.state.add_prompt(prompt)
    manager._check_rate_limited_prompts()

    assert prompt.status == PromptStatus.QUEUED


def test_manager_add_prompt_saves_to_disk(manager):  # QMG-024
    """QueueManager.add_prompt() saves the prompt to storage so it survives reload."""
    p = QueuedPrompt(content="new task via manager")
    manager.state = None

    result = manager.add_prompt(p)
    assert result is True

    reloaded = manager.storage.load_queue_state()
    ids = [pr.id for pr in reloaded.prompts]
    assert p.id in ids, (
        f"Prompt {p.id!r} not found in reloaded state. Found ids: {ids}"
    )


# ===========================================================================
# Non-Retryable Error Handling
# ===========================================================================


def test_non_retryable_error_immediately_fails_with_unlimited_retries(manager, mocker):  # QMG-NR-001
    """A non-retryable error marks the prompt FAILED immediately even when max_retries=-1."""
    prompt = QueuedPrompt(id="test01", content="task", max_retries=-1)
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    mocker.patch.object(
        manager.claude_interface, "execute_prompt", return_value=_non_retryable_result()
    )
    manager._execute_prompt(prompt)

    assert prompt.status == PromptStatus.FAILED
    assert prompt.retry_count == 0  # retry_count not incremented for non-retryable errors
    assert manager.state.failed_count == 1


def test_non_retryable_error_immediately_fails_with_finite_retries(manager, mocker):  # QMG-NR-002
    """A non-retryable error fails the prompt immediately even when retries remain."""
    prompt = QueuedPrompt(id="test02", content="task", max_retries=3)
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    mocker.patch.object(
        manager.claude_interface, "execute_prompt", return_value=_non_retryable_result()
    )
    manager._execute_prompt(prompt)

    assert prompt.status == PromptStatus.FAILED
    # retry_count not consumed: if re-queued manually, full budget is available
    assert prompt.retry_count == 0
    assert manager.state.failed_count == 1


def test_non_retryable_error_logged_as_non_retryable(manager, mocker):  # QMG-NR-003
    """Execution log mentions non-retryable so operator knows why no retry occurred."""
    prompt = QueuedPrompt(id="test03", content="task", max_retries=-1)
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    mocker.patch.object(
        manager.claude_interface, "execute_prompt", return_value=_non_retryable_result()
    )
    manager._execute_prompt(prompt)

    assert "non-retryable" in prompt.execution_log.lower()


def test_ordinary_failure_still_retries(manager, mocker):  # QMG-NR-004
    """A normal (retryable) failure still uses the can_retry() path."""
    prompt = QueuedPrompt(id="test04", content="task", max_retries=3)
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    mocker.patch.object(
        manager.claude_interface, "execute_prompt",
        return_value=ExecutionResult(success=False, output="", error="transient", execution_time=0.1)
    )
    manager._execute_prompt(prompt)

    assert prompt.status == PromptStatus.QUEUED  # retried, not failed
    assert prompt.retry_count == 1


# ===========================================================================
# Fix 3 — Retry Backoff for Generic Failures
# ===========================================================================


def test_generic_failure_sets_retry_not_before(manager):  # QMG-025
    """After a generic (non-rate-limited) failure with retries remaining,
    prompt.retry_not_before is set to a future datetime.
    """
    fail = _fail_result("transient error")
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="retry me", max_retries=3, retry_count=0)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)
    manager.state = None

    with patch.object(manager.claude_interface, "execute_prompt", return_value=fail):
        manager._process_queue_iteration()

    updated = next(pr for pr in manager.state.prompts if pr.id == p.id)
    assert updated.retry_not_before is not None
    assert updated.retry_not_before > datetime.now()


def test_prompt_with_future_retry_not_before_is_skipped(manager):  # QMG-026
    """A QUEUED prompt whose retry_not_before is in the future must not be
    selected for execution.
    """
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="not yet", max_retries=3, retry_count=1)
    p.retry_not_before = datetime.now() + timedelta(minutes=5)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)
    manager.state = None

    with patch.object(manager.claude_interface, "execute_prompt") as mock_exec:
        did_work = manager._process_queue_iteration()

    assert did_work is False
    mock_exec.assert_not_called()


def test_prompt_with_past_retry_not_before_is_eligible(manager):  # QMG-027
    """A QUEUED prompt whose retry_not_before is in the past must be eligible
    for execution.
    """
    success = _success_result()
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="ready now", max_retries=3, retry_count=1)
    p.retry_not_before = datetime.now() - timedelta(seconds=1)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)
    manager.state = None

    with patch.object(manager.claude_interface, "execute_prompt", return_value=success):
        did_work = manager._process_queue_iteration()

    assert did_work is True


def test_retry_not_before_not_set_on_final_failure(manager):  # QMG-028
    """When max retries are exhausted, the prompt becomes FAILED, not QUEUED.
    retry_not_before is irrelevant for FAILED prompts.
    """
    fail = _fail_result("final")
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="exhausted", max_retries=1, retry_count=1)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)
    manager.state = None

    with patch.object(manager.claude_interface, "execute_prompt", return_value=fail):
        manager._process_queue_iteration()

    updated = next(pr for pr in manager.state.prompts if pr.id == p.id)
    assert updated.status == PromptStatus.FAILED


def test_retry_not_before_persists_through_reload(manager):  # QMG-029
    """retry_not_before written to YAML frontmatter survives a state reload."""
    future = datetime.now() + timedelta(minutes=10)
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="persist test", max_retries=3, retry_count=1)
    p.retry_not_before = future
    state.add_prompt(p)
    manager.storage.save_queue_state(state)

    reloaded = manager.storage.load_queue_state()
    reloaded_prompt = next(pr for pr in reloaded.prompts if pr.id == p.id)
    assert reloaded_prompt.retry_not_before is not None
    delta = abs((reloaded_prompt.retry_not_before - future).total_seconds())
    assert delta < 5, f"Persisted retry_not_before drifted by {delta:.3f}s"
    assert reloaded_prompt.retry_not_before.tzinfo is None


def test_retry_not_before_does_not_affect_rate_limited_prompts(manager):  # QMG-030
    """A RATE_LIMITED prompt must be handled by reset_time / rate_limited_at,
    not by retry_not_before. The two mechanisms must not interfere.
    """
    prompt = QueuedPrompt(
        content="rate limited", status=PromptStatus.RATE_LIMITED, max_retries=3
    )
    prompt.rate_limited_at = datetime.now() - timedelta(minutes=6)
    prompt.reset_time = None
    prompt.retry_not_before = datetime.now() + timedelta(hours=1)  # future, but irrelevant
    prompt.retry_count = 0

    manager.state = QueueState()
    manager.state.add_prompt(prompt)
    manager._check_rate_limited_prompts()

    # 5-min heuristic should fire (rate_limited_at > 5min ago, no reset_time)
    assert prompt.status == PromptStatus.QUEUED
    assert prompt.should_execute_now() is True, (
        "Re-queued rate-limited prompt must be immediately selectable; "
        "retry_not_before must be cleared or already past"
    )


def test_cooldown_prompt_does_not_print_no_prompts_in_queue(manager, capsys):  # QMG-031
    """When a prompt is in retry_not_before cooldown and no other prompts exist,
    the iteration must NOT print "No prompts in queue". It must print a cooldown
    message that includes the remaining wait time.
    """
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="cooling down", max_retries=3, retry_count=1)
    p.retry_not_before = datetime.now() + timedelta(minutes=5)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)
    manager.state = None

    manager._process_queue_iteration()

    captured = capsys.readouterr()
    assert "No prompts in queue" not in captured.out, (
        "Must not print 'No prompts in queue' when a prompt is in cooldown"
    )
    assert "retry cooldown" in captured.out.lower(), (
        "Must print a message indicating the prompt is in retry cooldown"
    )


# ===========================================================================
# Interrupt / Signal Handling (QMG-INT)
# ===========================================================================


def test_signal_handler_calls_kill_current_before_stop(manager):  # QMG-INT-001
    """_signal_handler() calls kill_current() before stop() (verify ordering)."""
    call_log = []

    manager.claude_interface.kill_current = MagicMock(
        side_effect=lambda: call_log.append("kill_current")
    )
    original_stop = manager.stop
    manager.stop = MagicMock(
        side_effect=lambda: (call_log.append("stop"), original_stop())
    )

    manager._signal_handler(signal.SIGINT, None)

    assert call_log == ["kill_current", "stop"], (
        f"Expected kill_current before stop, got {call_log}"
    )


def test_keyboard_interrupt_bypasses_process_execution_result(manager, mocker):  # QMG-INT-002
    """When execute_prompt() raises KeyboardInterrupt, _process_execution_result()
    is bypassed and the prompt remains EXECUTING.
    """
    prompt = QueuedPrompt(id="test01", content="task")
    manager.state = manager.storage.load_queue_state()
    manager.state.add_prompt(prompt)

    mocker.patch.object(
        manager.claude_interface, "execute_prompt",
        side_effect=KeyboardInterrupt
    )

    with pytest.raises(KeyboardInterrupt):
        manager._execute_prompt(prompt)

    assert prompt.status == PromptStatus.EXECUTING


def test_end_to_end_signal_shutdown_reverts_prompt(manager, mocker):  # QMG-INT-003
    """End-to-end: signal → kill_current → KeyboardInterrupt → _shutdown() → QUEUED.

    WARNING: This test exercises true cross-thread access to _current_process
    and _interrupted WITHOUT locks.  This is safe under the GIL (CPython) but
    will race under free-threaded Python (PEP 703, --disable-gil).  If
    free-threaded builds become a target, the production code must add
    synchronization (not just this test).
    """
    prompt = QueuedPrompt(id="test02", content="task")
    state = manager.storage.load_queue_state()
    state.add_prompt(prompt)
    manager.storage.save_queue_state(state)
    manager.state = None

    # Event to synchronize the mock communicate() with the signal handler
    communicate_started = threading.Event()
    signal_sent = threading.Event()

    def mock_communicate(timeout=None):
        communicate_started.set()
        signal_sent.wait(timeout=5)
        # After signal handler ran, _interrupted should be True
        return ("output", "")

    mock_proc = MagicMock()
    mock_proc.pid = 99999
    mock_proc.communicate.side_effect = mock_communicate
    mock_proc.returncode = 0
    mock_proc.wait.return_value = 0

    execution_error = []

    def run_iteration():
        try:
            manager._process_queue_iteration()
        except KeyboardInterrupt:
            execution_error.append("KeyboardInterrupt")

    mocker.patch("subprocess.Popen", return_value=mock_proc)
    mocker.patch.object(ClaudeCodeInterface, "_kill_proc_group", return_value=True)

    worker = threading.Thread(target=run_iteration)
    worker.start()

    # Wait for communicate() to start, then send signal
    assert communicate_started.wait(timeout=5), "communicate() did not start"
    manager._signal_handler(signal.SIGINT, None)
    signal_sent.set()

    worker.join(timeout=5)
    assert not worker.is_alive(), "Worker thread did not finish"
    assert "KeyboardInterrupt" in execution_error

    # Verify _shutdown() reverts prompt to QUEUED
    manager._shutdown()

    reloaded = manager.storage.load_queue_state()
    prompt_after = next(
        (pr for pr in reloaded.prompts if pr.id == "test02"), None
    )
    assert prompt_after is not None, "Prompt must survive shutdown"
    assert prompt_after.status == PromptStatus.QUEUED, (
        f"Expected QUEUED after shutdown, got {prompt_after.status}"
    )

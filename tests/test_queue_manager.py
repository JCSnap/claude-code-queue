"""
Tests for queue_manager.py — QueueManager (ClaudeCodeInterface mocked).

ClaudeCodeInterface.execute_prompt is patched for all execution tests.
The manager fixture uses tmp_path so every test gets a fresh storage dir.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

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
    """A prompt that always fails is retried up to max_retries times then FAILED."""
    fail = _fail_result("build failed")
    state = manager.storage.load_queue_state()
    p = QueuedPrompt(content="failing task", max_retries=2)
    state.add_prompt(p)
    manager.storage.save_queue_state(state)

    with patch.object(manager.claude_interface, "execute_prompt", return_value=fail):
        for _ in range(5):
            manager.state = None
            manager._process_queue_iteration()

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

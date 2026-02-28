"""
Tests for models.py — QueuedPrompt, RateLimitInfo, QueueState.
"""

import re
from datetime import datetime, timedelta

import pytest

from claude_code_queue.models import (
    ExecutionResult,
    PromptStatus,
    QueuedPrompt,
    QueueState,
    RateLimitInfo,
)


# ===========================================================================
# QueuedPrompt — basic properties
# ===========================================================================


def test_prompt_id_is_8_hex_chars():  # MOD-001
    """Auto-generated id is 8 characters from the uuid4 hex."""
    p = QueuedPrompt(content="test")
    assert len(p.id) == 8
    assert all(c in "0123456789abcdef-" for c in p.id)


def test_add_log_appends_timestamped_entry():  # MOD-002
    """add_log() produces a line with [YYYY-MM-DD HH:MM:SS] prefix."""
    p = QueuedPrompt(content="test")
    p.add_log("step completed")
    assert "step completed" in p.execution_log
    assert re.search(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", p.execution_log
    ), "Log entry must have [YYYY-MM-DD HH:MM:SS] timestamp"


def test_add_log_is_cumulative():  # MOD-003
    """Multiple add_log() calls accumulate all entries."""
    p = QueuedPrompt(content="test")
    p.add_log("first")
    p.add_log("second")
    p.add_log("third")
    assert "first" in p.execution_log
    assert "second" in p.execution_log
    assert "third" in p.execution_log


# ===========================================================================
# QueuedPrompt — should_execute_now()
# ===========================================================================


def test_should_execute_now_queued_returns_true():  # MOD-004
    """QUEUED status always passes the rate-limit check."""
    p = QueuedPrompt(content="test", status=PromptStatus.QUEUED)
    assert p.should_execute_now() is True


def test_should_execute_now_executing_returns_true():  # MOD-005
    """EXECUTING status also returns True — method is not a state-machine guard."""
    p = QueuedPrompt(content="test", status=PromptStatus.EXECUTING)
    assert p.should_execute_now() is True


def test_should_execute_now_rate_limited_no_reset_time_returns_false():  # MOD-006
    """RATE_LIMITED with no reset_time: we don't know the window — stay blocked."""
    p = QueuedPrompt(content="test", status=PromptStatus.RATE_LIMITED)
    p.reset_time = None
    assert p.should_execute_now() is False


def test_should_execute_now_rate_limited_past_reset_time_returns_true():  # MOD-007
    """RATE_LIMITED whose reset window has passed: allow execution."""
    p = QueuedPrompt(content="test", status=PromptStatus.RATE_LIMITED)
    p.reset_time = datetime.now() - timedelta(minutes=5)
    assert p.should_execute_now() is True


def test_should_execute_now_rate_limited_future_reset_time_returns_false():  # MOD-008
    """RATE_LIMITED with a future reset window: stay blocked."""
    p = QueuedPrompt(content="test", status=PromptStatus.RATE_LIMITED)
    p.reset_time = datetime.now() + timedelta(hours=2)
    assert p.should_execute_now() is False


# ===========================================================================
# QueuedPrompt — can_retry()
# ===========================================================================


def test_can_retry_true_when_status_is_executing():  # MOD-009
    """can_retry() must not gate on status when EXECUTING."""
    prompt = QueuedPrompt(
        content="test",
        status=PromptStatus.EXECUTING,
        retry_count=0,
        max_retries=3,
    )
    assert prompt.can_retry() is True


def test_can_retry_true_when_status_is_failed():  # MOD-010
    """can_retry() works for FAILED status (FAILED is in the allowed list)."""
    prompt = QueuedPrompt(
        content="test",
        status=PromptStatus.FAILED,
        retry_count=0,
        max_retries=3,
    )
    assert prompt.can_retry() is True


def test_can_retry_false_when_retry_count_equals_max_retries():  # MOD-011
    """Counter boundary: retry_count == max_retries → cannot retry."""
    prompt = QueuedPrompt(
        content="test",
        status=PromptStatus.EXECUTING,
        retry_count=3,
        max_retries=3,
    )
    assert prompt.can_retry() is False


def test_can_retry_false_when_retry_count_exceeds_max_retries():  # MOD-012
    """Counter boundary: retry_count > max_retries → cannot retry."""
    prompt = QueuedPrompt(
        content="test",
        status=PromptStatus.EXECUTING,
        retry_count=5,
        max_retries=3,
    )
    assert prompt.can_retry() is False


def test_can_retry_unlimited_returns_true_at_any_count():  # MOD-013
    """max_retries=-1 means unlimited; 999 < -1 must NOT be the check."""
    prompt = QueuedPrompt(
        content="test",
        status=PromptStatus.EXECUTING,
        retry_count=999,
        max_retries=-1,
    )
    assert prompt.can_retry() is True


def test_max_retries_minus_one_never_false():  # MOD-014
    """Exhaustive: unlimited retries never returns False across 51 increments."""
    prompt = QueuedPrompt(
        content="test",
        status=PromptStatus.EXECUTING,
        max_retries=-1,
    )
    for count in range(51):
        prompt.retry_count = count
        assert prompt.can_retry() is True, (
            f"can_retry() returned False at retry_count={count} with max_retries=-1"
        )


def test_can_retry_max_retries_zero():  # MOD-015
    """Edge case: max_retries=0 means execute once and never retry."""
    prompt = QueuedPrompt(
        content="test",
        status=PromptStatus.EXECUTING,
        max_retries=0,
        retry_count=0,
    )
    assert prompt.can_retry() is False


def test_max_retries_3_total_attempts_semantics():  # MOD-016
    """max_retries=3 allows retry_count 0,1,2 → True; at 3 → False."""
    prompt = QueuedPrompt(
        content="test", status=PromptStatus.EXECUTING, max_retries=3
    )
    for count in range(3):
        prompt.retry_count = count
        assert prompt.can_retry() is True, f"Expected True at retry_count={count}"
    prompt.retry_count = 3
    assert prompt.can_retry() is False


def test_max_retries_1_means_no_retries():  # MOD-017
    """max_retries=1: after the single attempt (retry_count incremented to 1) → False."""
    prompt = QueuedPrompt(
        content="test",
        status=PromptStatus.EXECUTING,
        max_retries=1,
    )
    prompt.retry_count = 1
    assert prompt.can_retry() is False


# ===========================================================================
# QueueState — get_next_prompt, add/remove, stats
# ===========================================================================


def test_get_next_prompt_returns_lowest_priority_number():  # MOD-018
    """get_next_prompt() picks the prompt with the numerically smallest priority."""
    state = QueueState()
    state.add_prompt(QueuedPrompt(content="high", priority=2))
    state.add_prompt(QueuedPrompt(content="medium", priority=1))
    state.add_prompt(QueuedPrompt(content="low", priority=0))
    next_p = state.get_next_prompt()
    assert next_p is not None
    assert next_p.content == "low"
    assert next_p.priority == 0


def test_get_next_prompt_skips_non_queued():  # MOD-019
    """EXECUTING and COMPLETED prompts are invisible to get_next_prompt()."""
    state = QueueState()
    state.add_prompt(
        QueuedPrompt(content="executing", status=PromptStatus.EXECUTING)
    )
    state.add_prompt(
        QueuedPrompt(content="completed", status=PromptStatus.COMPLETED)
    )
    state.add_prompt(QueuedPrompt(content="queued", status=PromptStatus.QUEUED))
    next_p = state.get_next_prompt()
    assert next_p is not None
    assert next_p.content == "queued"


def test_get_next_prompt_returns_none_when_empty():  # MOD-020
    """Empty queue → get_next_prompt() returns None (no crash)."""
    state = QueueState()
    assert state.get_next_prompt() is None


def test_get_next_prompt_skips_rate_limited():  # MOD-021
    """A RATE_LIMITED prompt with a future reset_time must not be returned."""
    state = QueueState()
    p = QueuedPrompt(content="rate-limited", status=PromptStatus.RATE_LIMITED)
    p.reset_time = datetime.now() + timedelta(hours=2)
    state.add_prompt(p)
    assert state.get_next_prompt() is None


def test_get_next_prompt_promotes_rate_limited_when_reset_time_past():  # MOD-022
    """get_next_prompt() promotes a RATE_LIMITED prompt whose reset window has closed.

    The prompt's status is mutated to QUEUED in-place before being returned.
    """
    state = QueueState()
    p = QueuedPrompt(
        content="rate-limited-ready",
        status=PromptStatus.RATE_LIMITED,
        max_retries=3,
        retry_count=1,
    )
    p.reset_time = datetime.now() - timedelta(minutes=5)
    state.add_prompt(p)

    result = state.get_next_prompt()
    assert result is not None, "Should promote the rate-limited prompt"
    assert result.status == PromptStatus.QUEUED, "Promotion sets status to QUEUED in-place"
    assert result.id == p.id


def test_add_prompt_increments_count():  # MOD-023
    """add_prompt() appends exactly one item to state.prompts."""
    state = QueueState()
    assert len(state.prompts) == 0
    state.add_prompt(QueuedPrompt(content="test"))
    assert len(state.prompts) == 1


def test_remove_prompt_decrements_count():  # MOD-024
    """remove_prompt() eliminates the prompt with the matching id."""
    state = QueueState()
    p = QueuedPrompt(content="test")
    state.add_prompt(p)
    state.remove_prompt(p.id)
    assert all(pr.id != p.id for pr in state.prompts)


def test_get_stats_counts_by_status():  # MOD-025
    """get_stats() returns counts nested under 'status_counts'.

    - 'queued' counts active prompts in the in-memory list.
    - 'completed' and 'failed' use the persistent counters
      (total_processed / failed_count), NOT the prompt list.
    - The stats dict is nested: stats['status_counts']['queued'], etc.
    """
    state = QueueState()
    state.add_prompt(QueuedPrompt(content="q1", status=PromptStatus.QUEUED))
    state.add_prompt(QueuedPrompt(content="q2", status=PromptStatus.QUEUED))
    state.total_processed = 1
    state.failed_count = 1

    stats = state.get_stats()

    assert stats["status_counts"]["queued"] == 2
    assert stats["status_counts"]["completed"] == 1
    assert stats["status_counts"]["failed"] == 1
    assert stats["total_prompts"] == 2
    assert stats["total_processed"] == 1
    assert stats["failed_count"] == 1


# ===========================================================================
# RateLimitInfo — model-level tests
# ===========================================================================


def test_rate_limit_info_is_rate_limited_false_by_default():  # MOD-026
    """RateLimitInfo() constructor defaults: not rate-limited, no reset time."""
    info = RateLimitInfo()
    assert info.is_rate_limited is False
    assert info.reset_time is None


def test_from_claude_response_rate_limit_detected():  # MOD-027
    """RateLimitInfo.from_claude_response() detects the broad 'rate limit' phrase."""
    info = RateLimitInfo.from_claude_response("rate limit hit on this account")
    assert info.is_rate_limited is True
    assert info.limit_message != ""
    assert info.timestamp is not None


def test_from_claude_response_normal_not_rate_limited():  # MOD-028
    """from_claude_response() returns is_rate_limited=False for normal output."""
    info = RateLimitInfo.from_claude_response(
        "Successfully updated the authentication module."
    )
    assert info.is_rate_limited is False
    assert info.reset_time is None

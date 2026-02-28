"""
Tests for storage.py — MarkdownPromptParser, QueueStorage.
"""

import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claude_code_queue.models import (
    PromptStatus,
    QueuedPrompt,
    QueueState,
)
from claude_code_queue.storage import MarkdownPromptParser, QueueStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_FRONTMATTER = (
    "---\n"
    "priority: 0\n"
    "working_directory: .\n"
    "max_retries: 3\n"
    "status: queued\n"
    "retry_count: 0\n"
    "created_at: 2025-01-01T00:00:00\n"
    "---\n\n"
)


def _make_executing_file(queue_dir: Path, prompt_id: str, title: str, content: str) -> Path:
    """Write a .executing.md file as if a previous run crashed mid-execution."""
    path = queue_dir / f"{prompt_id}-{title}.executing.md"
    path.write_text(
        f"---\n"
        f"priority: 0\n"
        f"working_directory: .\n"
        f"max_retries: 3\n"
        f"status: executing\n"
        f"retry_count: 0\n"
        f"created_at: 2025-01-01T00:00:00\n"
        f"---\n\n"
        f"{content}"
    )
    return path


# ===========================================================================
# Orphan Recovery
# ===========================================================================


def test_queued_transition_removes_stale_executing_file(tmp_path):  # STO-001
    """Saving a QUEUED prompt must delete any stale .executing.md file."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(id="abc12345", content="test prompt")

    stale_path = storage.queue_dir / "abc12345-test-prompt.executing.md"
    stale_path.write_text(MINIMAL_FRONTMATTER + "test prompt")
    assert stale_path.exists(), "Stale file must exist before the action"

    prompt.status = PromptStatus.QUEUED
    storage._save_single_prompt(prompt)

    assert not stale_path.exists(), "Stale .executing.md was not removed"
    regular_files = list(storage.queue_dir.glob("abc12345-*.md"))
    non_executing = [f for f in regular_files if not f.name.endswith(".executing.md")]
    assert len(non_executing) == 1, (
        f"Expected one plain .md for the prompt, found: {regular_files}"
    )
    executing_files = list(storage.queue_dir.glob("abc12345-*.executing.md"))
    assert len(executing_files) == 0, "No .executing.md should remain"


def test_load_executing_file_returns_queued_status(tmp_path):  # STO-002
    """.executing.md files must be recovered as QUEUED (at-least-once semantics)."""
    storage = QueueStorage(str(tmp_path))
    _make_executing_file(storage.queue_dir, "abc12345", "my-prompt", "my prompt content")

    state = storage.load_queue_state()

    assert len(state.prompts) == 1
    assert state.prompts[0].status == PromptStatus.QUEUED
    assert state.prompts[0].id == "abc12345"


def test_load_executing_file_adds_recovery_log(tmp_path):  # STO-003
    """Recovered prompts must have a log entry noting the recovery."""
    storage = QueueStorage(str(tmp_path))
    _make_executing_file(storage.queue_dir, "abc12345", "my-prompt", "my prompt content")

    state = storage.load_queue_state()

    assert len(state.prompts) == 1
    log = state.prompts[0].execution_log
    assert "recovered" in log.lower(), (
        f"Expected 'Recovered' in execution_log, got: {log!r}"
    )


def test_load_executing_does_not_leave_executing_status(tmp_path):  # STO-004
    """No prompt should have EXECUTING status after load_queue_state()."""
    storage = QueueStorage(str(tmp_path))
    _make_executing_file(storage.queue_dir, "abc12345", "crashed", "crashed content")
    (storage.queue_dir / "def67890-normal.md").write_text(
        MINIMAL_FRONTMATTER + "normal content"
    )

    state = storage.load_queue_state()

    assert all(p.status != PromptStatus.EXECUTING for p in state.prompts), (
        f"Found EXECUTING prompt(s) after load: "
        f"{[p.id for p in state.prompts if p.status == PromptStatus.EXECUTING]}"
    )


# ===========================================================================
# Content Corruption / Sentinel
# ===========================================================================


def test_write_prompt_file_includes_execution_log_section(tmp_path):  # STO-005
    """write_prompt_file() must write '## Execution Log' before the log body
    so the parser can cleanly split the two sections.
    """
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(id="abc12345", content="my task")
    prompt.add_log("Step 1 done")
    file_path = storage.queue_dir / "abc12345-my-task.md"
    storage.parser.write_prompt_file(prompt, file_path)
    raw = file_path.read_text()
    assert "## Execution Log" in raw


def test_parse_strips_sentinel_log_from_content(tmp_path):  # STO-006
    """parse_prompt_file() must not include the sentinel or log in content."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-my-task.md"
    file_path.write_text(
        "---\npriority: 0\nworking_directory: .\nmax_retries: 3\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\n"
        "my task content\n\n"
        "<!-- claude-queue:execution-log -->\n"
        "```\n[2025-01-01 12:00:00] Step 1 done\n```\n"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert "<!-- claude-queue:execution-log -->" not in prompt.content
    assert "Step 1 done" not in prompt.content
    assert prompt.content.strip() == "my task content"


def test_parse_strips_legacy_execution_log_from_content(tmp_path):  # STO-007
    """Legacy '## Execution Log' separator must also be stripped
    (backwards compatibility with files written before the sentinel existed).
    """
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-my-task.md"
    file_path.write_text(
        "---\npriority: 0\nworking_directory: .\nmax_retries: 3\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\n"
        "original content\n\n"
        "## Execution Log\n\n"
        "```\n[2025-01-01 12:00:00] Some log entry\n```\n"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert "## Execution Log" not in prompt.content
    assert "Some log entry" not in prompt.content
    assert prompt.content.strip() == "original content"


def test_content_survives_write_parse_roundtrip_with_log(tmp_path):  # STO-008
    """Content is unchanged after write → add log → write again → parse."""
    storage = QueueStorage(str(tmp_path))
    original_content = "Fix the authentication module and add unit tests."
    prompt = QueuedPrompt(id="abc12345", content=original_content)
    file_path = storage.queue_dir / "abc12345-fix-the-authentication.md"
    storage.parser.write_prompt_file(prompt, file_path)
    prompt.add_log("First attempt failed")
    storage.parser.write_prompt_file(prompt, file_path)
    parsed = storage.parser.parse_prompt_file(file_path)
    assert parsed is not None
    assert parsed.content == original_content


def test_content_without_log_unchanged(tmp_path):  # STO-009
    """Write/parse with no log entries: content must be identical to original."""
    storage = QueueStorage(str(tmp_path))
    content = "Simple task with no log entries."
    prompt = QueuedPrompt(id="abc12345", content=content)
    file_path = storage.queue_dir / "abc12345-simple-task.md"
    storage.parser.write_prompt_file(prompt, file_path)
    parsed = storage.parser.parse_prompt_file(file_path)
    assert parsed is not None
    assert parsed.content == content


def test_log_section_present_in_raw_file(tmp_path):  # STO-010
    """Even if the log is stripped from content on parse, it must be on disk."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(id="abc12345", content="task")
    prompt.add_log("Execution started")
    file_path = storage.queue_dir / "abc12345-task.md"
    storage.parser.write_prompt_file(prompt, file_path)
    raw = file_path.read_text()
    assert "Execution started" in raw


# ===========================================================================
# Rate-Limit Persistence
# ===========================================================================


def test_retry_count_persisted_and_reloaded(tmp_path):  # STO-011
    """retry_count is written to frontmatter and reloaded correctly."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(id="abc12345", content="task")
    prompt.retry_count = 2
    state = QueueState()
    state.add_prompt(prompt)
    storage.save_queue_state(state)
    reloaded = storage.load_queue_state()
    assert reloaded.prompts[0].retry_count == 2


def test_rate_limited_at_persisted_and_reloaded(tmp_path):  # STO-012
    """rate_limited_at datetime survives a save/load cycle."""
    storage = QueueStorage(str(tmp_path))
    ts = datetime(2025, 1, 1, 12, 0, 0)
    prompt = QueuedPrompt(id="abc12345", content="task")
    prompt.status = PromptStatus.RATE_LIMITED
    prompt.rate_limited_at = ts
    state = QueueState()
    state.add_prompt(prompt)
    storage.save_queue_state(state)
    reloaded = storage.load_queue_state()
    assert reloaded.prompts[0].rate_limited_at is not None
    delta = abs(
        (reloaded.prompts[0].rate_limited_at - ts).total_seconds()
    )
    assert delta < 1, (
        f"rate_limited_at drifted by {delta:.3f}s; expected < 1s"
    )


def test_reset_time_persisted_and_reloaded(tmp_path):  # STO-013
    """reset_time datetime survives a save/load cycle."""
    storage = QueueStorage(str(tmp_path))
    ts = datetime(2025, 6, 1, 15, 0, 0)
    prompt = QueuedPrompt(id="abc12345", content="task")
    prompt.status = PromptStatus.RATE_LIMITED
    prompt.reset_time = ts
    state = QueueState()
    state.add_prompt(prompt)
    storage.save_queue_state(state)
    reloaded = storage.load_queue_state()
    assert reloaded.prompts[0].reset_time is not None
    delta = abs((reloaded.prompts[0].reset_time - ts).total_seconds())
    assert delta < 1, f"reset_time drifted by {delta:.3f}s; expected < 1s"


def test_last_executed_persisted_and_reloaded(tmp_path):  # STO-014
    """last_executed datetime survives a save/load cycle."""
    storage = QueueStorage(str(tmp_path))
    ts = datetime(2025, 3, 15, 9, 30, 0)
    prompt = QueuedPrompt(id="abc12345", content="task")
    prompt.last_executed = ts
    state = QueueState()
    state.add_prompt(prompt)
    storage.save_queue_state(state)
    reloaded = storage.load_queue_state()
    assert reloaded.prompts[0].last_executed is not None
    delta = abs((reloaded.prompts[0].last_executed - ts).total_seconds())
    assert delta < 1, f"last_executed drifted by {delta:.3f}s; expected < 1s"


def test_parse_optional_datetime_handles_timezone_aware(tmp_path):  # STO-015
    """_parse_optional_datetime() strips tzinfo to produce a naive datetime."""
    result = QueueStorage._parse_optional_datetime("2025-01-01T12:00:00+00:00")
    assert result is not None
    assert result.tzinfo is None, "Must return naive datetime (no tzinfo)"
    assert result.year == 2025
    assert result.hour == 12


def test_parse_optional_datetime_handles_none(tmp_path):  # STO-016
    """_parse_optional_datetime(None) returns None without raising."""
    result = QueueStorage._parse_optional_datetime(None)
    assert result is None


def test_parse_optional_datetime_handles_invalid_string(tmp_path):  # STO-017
    """_parse_optional_datetime('not-a-date') returns None without raising."""
    result = QueueStorage._parse_optional_datetime("not-a-date")
    assert result is None


# ===========================================================================
# MarkdownPromptParser — frontmatter field reading
# ===========================================================================


def test_parse_reads_priority(tmp_path):  # STO-018
    """priority: 2 in frontmatter → prompt.priority == 2."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\npriority: 2\nworking_directory: .\nmax_retries: 3\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\nmy task"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert prompt.priority == 2


def test_parse_reads_working_directory(tmp_path):  # STO-019
    """working_directory: /tmp/myproject in frontmatter is faithfully loaded."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\npriority: 0\nworking_directory: /tmp/myproject\nmax_retries: 3\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\ncontent"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert prompt.working_directory == "/tmp/myproject"


def test_parse_reads_context_files_as_list(tmp_path):  # STO-020
    """context_files: [a.py, b.py] in frontmatter → list of two strings."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\npriority: 0\nworking_directory: .\nmax_retries: 3\n"
        "context_files:\n- a.py\n- b.py\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\ncontent"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert prompt.context_files == ["a.py", "b.py"]


def test_parse_reads_max_retries(tmp_path):  # STO-021
    """max_retries: 5 in frontmatter → prompt.max_retries == 5."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\npriority: 0\nworking_directory: .\nmax_retries: 5\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\ncontent"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert prompt.max_retries == 5


def test_parse_reads_estimated_tokens(tmp_path):  # STO-022
    """estimated_tokens: 1500 in frontmatter → prompt.estimated_tokens == 1500."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\npriority: 0\nworking_directory: .\nmax_retries: 3\n"
        "estimated_tokens: 1500\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\ncontent"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert prompt.estimated_tokens == 1500


def test_parse_estimated_tokens_null(tmp_path):  # STO-023
    """estimated_tokens: null in frontmatter → prompt.estimated_tokens is None."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\npriority: 0\nworking_directory: .\nmax_retries: 3\n"
        "estimated_tokens: null\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\ncontent"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert prompt.estimated_tokens is None


def test_parse_defaults_when_keys_missing(tmp_path):  # STO-024
    """Minimal frontmatter → defaults: priority=0, max_retries=3, context_files=[],
    estimated_tokens=None.
    """
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\nworking_directory: .\n---\ncontent here"
    )
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert prompt.priority == 0
    assert prompt.max_retries == 3
    assert prompt.context_files == []
    assert prompt.estimated_tokens is None


def test_parse_file_without_frontmatter(tmp_path):  # STO-025
    """A file with no '---' block must not crash; content is the entire file."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-plain.md"
    file_path.write_text("Just plain content, no frontmatter at all.")
    prompt = storage.parser.parse_prompt_file(file_path)
    assert prompt is not None
    assert "plain content" in prompt.content
    assert prompt.priority == 0


def test_write_then_parse_roundtrip_preserves_priority(tmp_path):  # STO-026
    """Write with priority=7 → parse back → priority is still 7."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(id="abc12345", content="task", priority=7)
    file_path = storage.queue_dir / "abc12345-task.md"
    storage.parser.write_prompt_file(prompt, file_path)
    parsed = storage.parser.parse_prompt_file(file_path)
    assert parsed is not None
    assert parsed.priority == 7


# ===========================================================================
# Filename Generation
# ===========================================================================


def test_get_base_filename_sanitizes_special_chars():  # STO-027
    """Special characters in content are removed from the generated filename."""
    prompt = QueuedPrompt(
        id="abc12345", content="Fix <auth> module: add/remove tokens?"
    )
    filename = MarkdownPromptParser.get_base_filename(prompt)
    assert "/" not in filename
    assert "<" not in filename
    assert ">" not in filename
    assert "?" not in filename
    assert filename.startswith("abc12345-")


def test_get_base_filename_format():  # STO-028
    """Generated filename starts with '{id}-' and ends with '.md'."""
    prompt = QueuedPrompt(id="abc12345", content="some task description")
    filename = MarkdownPromptParser.get_base_filename(prompt)
    assert filename.startswith("abc12345-")
    assert filename.endswith(".md")


def test_get_base_filename_empty_content_produces_no_title():  # STO-029
    """QueuedPrompt with empty content → filename ends with '-no-title.md'."""
    prompt = QueuedPrompt(id="abc12345", content="")
    filename = MarkdownPromptParser.get_base_filename(prompt)
    assert filename.endswith("-no-title.md"), (
        f"Expected filename to end with '-no-title.md', got: {filename!r}"
    )
    assert "--" not in filename.replace("no-title", ""), (
        "Filename must not contain double-dash before .md"
    )


def test_get_base_filename_whitespace_only_produces_no_title():  # STO-030
    """Content that sanitizes to empty string (whitespace only) → 'no-title'."""
    prompt = QueuedPrompt(id="abc12345", content="   ")
    filename = MarkdownPromptParser.get_base_filename(prompt)
    assert "no-title" in filename


def test_get_base_filename_normal_content_unchanged():  # STO-031
    """Non-empty content still produces a normal filename."""
    prompt = QueuedPrompt(id="abc12345", content="Fix the auth module")
    filename = MarkdownPromptParser.get_base_filename(prompt)
    assert filename.startswith("abc12345-")
    assert "no-title" not in filename
    assert filename.endswith(".md")


def test_sanitize_filename_truncates_at_50_chars():  # STO-032
    """Pure alpha text of 100 chars is truncated to exactly 50 chars."""
    long_text = "a" * 100
    result = QueueStorage._sanitize_filename_static(long_text)
    assert len(result) <= 50
    assert result == "a" * 50


def test_sanitize_filename_collapses_consecutive_dashes():  # STO-033
    """Multiple consecutive dashes and whitespace are collapsed to a single '-'."""
    result = QueueStorage._sanitize_filename_static("fix---this   issue")
    assert "--" not in result, "Consecutive dashes must be collapsed"
    assert " " not in result, "Spaces must be replaced"
    assert result == "fix-this-issue"


# ===========================================================================
# QueueStorage — file placement by status
# ===========================================================================


def test_add_prompt_creates_file_in_queue_dir(tmp_path):  # STO-034
    """save_queue_state() with one QUEUED prompt → exactly one .md in queue_dir."""
    storage = QueueStorage(str(tmp_path))
    state = QueueState()
    prompt = QueuedPrompt(content="task one")
    state.add_prompt(prompt)
    storage.save_queue_state(state)
    files = list(storage.queue_dir.glob("*.md"))
    assert len(files) == 1


def test_load_queue_state_finds_added_prompt(tmp_path):  # STO-035
    """After save_queue_state(), load_queue_state() returns the same prompt."""
    storage = QueueStorage(str(tmp_path))
    state = QueueState()
    prompt = QueuedPrompt(id="abc12345", content="remembered task")
    state.add_prompt(prompt)
    storage.save_queue_state(state)
    reloaded = storage.load_queue_state()
    ids = [p.id for p in reloaded.prompts]
    assert "abc12345" in ids


def test_executing_file_has_executing_extension(tmp_path):  # STO-036
    """EXECUTING prompt → file gets .executing.md double extension."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(content="task", status=PromptStatus.EXECUTING)
    storage._save_single_prompt(prompt)
    executing_files = list(storage.queue_dir.glob("*.executing.md"))
    assert len(executing_files) == 1


def test_rate_limited_file_has_rate_limited_extension(tmp_path):  # STO-037
    """RATE_LIMITED prompt → file gets .rate-limited.md double extension."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(content="task", status=PromptStatus.RATE_LIMITED)
    storage._save_single_prompt(prompt)
    rl_files = list(storage.queue_dir.glob("*.rate-limited.md"))
    assert len(rl_files) == 1


def test_completed_prompt_moves_to_completed_dir(tmp_path):  # STO-038
    """COMPLETED prompt → file in completed_dir, nothing left in queue_dir."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(content="task", status=PromptStatus.COMPLETED)
    storage._save_single_prompt(prompt)
    files = list(storage.completed_dir.glob("*.md"))
    assert len(files) == 1
    queue_files = list(storage.queue_dir.glob(f"{prompt.id}*.md"))
    assert len(queue_files) == 0


def test_failed_prompt_moves_to_failed_dir(tmp_path):  # STO-039
    """FAILED prompt → file in failed_dir, nothing left in queue_dir."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(content="task", status=PromptStatus.FAILED)
    storage._save_single_prompt(prompt)
    files = list(storage.failed_dir.glob("*.md"))
    assert len(files) == 1
    queue_files = list(storage.queue_dir.glob(f"{prompt.id}*.md"))
    assert len(queue_files) == 0


def test_cancelled_prompt_naming(tmp_path):  # STO-040
    """CANCELLED prompt → file in failed_dir with 'cancelled' in name."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(id="abc12345", content="task", status=PromptStatus.CANCELLED)
    storage._save_single_prompt(prompt)
    files = list(storage.failed_dir.glob("*cancelled*"))
    assert len(files) == 1


# ===========================================================================
# Template Operations
# ===========================================================================


def test_create_prompt_template_writes_markdown(tmp_path):  # STO-041
    """create_prompt_template() creates a file that starts with '---'."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.create_prompt_template("my-task")
    assert file_path.exists()
    content = file_path.read_text()
    assert content.startswith("---")


def test_template_contains_required_frontmatter_keys(tmp_path):  # STO-042
    """Template file includes priority, working_directory, context_files, max_retries."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.create_prompt_template("my-task")
    content = file_path.read_text()
    assert "priority" in content
    assert "working_directory" in content
    assert "context_files" in content
    assert "max_retries" in content


def test_add_prompt_from_markdown_file(tmp_path):  # STO-043
    """add_prompt_from_markdown() parses an existing file and returns a QueuedPrompt."""
    storage = QueueStorage(str(tmp_path))
    external_file = tmp_path / "my-prompt.md"
    external_file.write_text(
        "---\npriority: 1\nworking_directory: /tmp\nmax_retries: 3\n"
        "status: queued\nretry_count: 0\ncreated_at: 2025-01-01T00:00:00\n---\n\n"
        "Do something useful."
    )
    prompt = storage.add_prompt_from_markdown(external_file)
    assert prompt is not None
    assert "Do something useful." in prompt.content
    assert prompt.status == PromptStatus.QUEUED


# ===========================================================================
# Storage Utility Methods
# ===========================================================================


def test_remove_prompt_files_clears_all_status_variants(tmp_path):  # STO-044
    """_remove_prompt_files() deletes the plain, .executing, and .rate-limited variants."""
    storage = QueueStorage(str(tmp_path))
    prompt_id = "abc12345"
    (storage.queue_dir / f"{prompt_id}-task.md").write_text("content")
    (storage.queue_dir / f"{prompt_id}-task.executing.md").write_text("content")
    (storage.queue_dir / f"{prompt_id}-task.rate-limited.md").write_text("content")

    storage._remove_prompt_files(prompt_id, storage.queue_dir)

    remaining = list(storage.queue_dir.glob(f"{prompt_id}*.md"))
    assert len(remaining) == 0, f"Files still present: {remaining}"


def test_state_stats_persist_across_sessions(tmp_path):  # STO-045
    """total_processed, failed_count, rate_limited_count survive a save/load cycle."""
    storage = QueueStorage(str(tmp_path))
    state = QueueState()
    state.total_processed = 5
    state.failed_count = 2
    state.rate_limited_count = 1
    storage.save_queue_state(state)

    reloaded = storage.load_queue_state()
    assert reloaded.total_processed == 5
    assert reloaded.failed_count == 2
    assert reloaded.rate_limited_count == 1


def test_load_state_without_json_file_returns_defaults(tmp_path):  # STO-046
    """When no queue-state.json exists, load_queue_state() returns zero counters."""
    storage = QueueStorage(str(tmp_path))
    assert not storage.state_file.exists()
    state = storage.load_queue_state()
    assert state.total_processed == 0
    assert state.failed_count == 0


def test_load_state_with_corrupt_json_returns_defaults(tmp_path):  # STO-047
    """Corrupt JSON file does not crash; load_queue_state() returns defaults."""
    storage = QueueStorage(str(tmp_path))
    storage.state_file.write_text("{ this is not valid json }")
    state = storage.load_queue_state()
    assert state.total_processed == 0


# ===========================================================================
# YAML Error Handling
# ===========================================================================


def test_yaml_parse_error_emits_warning_to_stderr(tmp_path, capsys):  # STO-048
    """Broken YAML frontmatter → warning on stderr; function does not raise."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\n"
        "priority: [unclosed bracket\n"
        "---\n\n"
        "task content"
    )
    result = storage.parser.parse_prompt_file(file_path)

    assert result is not None, "parse_prompt_file must not return None for YAML errors"
    captured = capsys.readouterr()
    assert "Warning" in captured.err or "warning" in captured.err.lower(), (
        f"Expected a warning on stderr, got: {captured.err!r}"
    )


def test_yaml_parse_error_uses_default_values(tmp_path):  # STO-049
    """Broken YAML frontmatter → returned prompt has default field values."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.queue_dir / "abc12345-task.md"
    file_path.write_text(
        "---\n"
        "priority: [unclosed bracket\n"
        "---\n\n"
        "task content"
    )
    result = storage.parser.parse_prompt_file(file_path)

    assert result is not None
    assert result.priority == 0
    assert result.max_retries == 3
    assert result.content == "task content"


# ===========================================================================
# File Permissions
# ===========================================================================


def test_storage_directories_have_mode_700(tmp_path):  # STO-050
    """All queue directories are created with mode 0o700."""
    storage = QueueStorage(str(tmp_path))
    for d in [storage.base_dir, storage.queue_dir, storage.completed_dir,
              storage.failed_dir, storage.bank_dir]:
        mode = stat.S_IMODE(d.stat().st_mode)
        assert mode == 0o700, (
            f"Expected mode 0o700 on {d}, got 0o{mode:o}"
        )


def test_write_prompt_file_has_mode_600(tmp_path):  # STO-051
    """write_prompt_file() creates a file with mode 0o600."""
    storage = QueueStorage(str(tmp_path))
    prompt = QueuedPrompt(id="abc12345", content="task")
    file_path = storage.queue_dir / "abc12345-task.md"
    storage.parser.write_prompt_file(prompt, file_path)
    mode = stat.S_IMODE(file_path.stat().st_mode)
    assert mode == 0o600, f"Expected mode 0o600, got 0o{mode:o}"


def test_create_prompt_template_has_mode_600(tmp_path):  # STO-052
    """create_prompt_template() creates a file with mode 0o600."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.create_prompt_template("my-task")
    mode = stat.S_IMODE(file_path.stat().st_mode)
    assert mode == 0o600, f"Expected mode 0o600, got 0o{mode:o}"


def test_save_prompt_to_bank_has_mode_600(tmp_path):  # STO-053
    """save_prompt_to_bank() creates a file with mode 0o600."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.save_prompt_to_bank("my-template")
    mode = stat.S_IMODE(file_path.stat().st_mode)
    assert mode == 0o600, f"Expected mode 0o600, got 0o{mode:o}"


# ===========================================================================
# Bank Operations
# ===========================================================================


def test_bank_save_creates_file_in_bank_dir(tmp_path):  # STO-054
    """save_prompt_to_bank() creates a file inside bank_dir."""
    storage = QueueStorage(str(tmp_path))
    file_path = storage.save_prompt_to_bank("daily-review")
    assert file_path.exists()
    assert file_path.parent == storage.bank_dir


def test_bank_list_returns_saved_template_names(tmp_path):  # STO-055
    """list_bank_templates() returns dicts containing the saved template names."""
    storage = QueueStorage(str(tmp_path))
    storage.save_prompt_to_bank("alpha")
    storage.save_prompt_to_bank("beta")
    templates = storage.list_bank_templates()
    names = [t["name"] for t in templates]
    assert "alpha" in names
    assert "beta" in names


def test_bank_list_empty_when_no_templates(tmp_path):  # STO-056
    """list_bank_templates() returns [] when the bank directory is empty."""
    storage = QueueStorage(str(tmp_path))
    assert storage.list_bank_templates() == []


def test_bank_use_copies_prompt_to_queue(tmp_path):  # STO-057
    """use_bank_template() returns a QueuedPrompt; saving it creates a queue file."""
    storage = QueueStorage(str(tmp_path))
    storage.save_prompt_to_bank("daily")
    (storage.bank_dir / "daily.md").write_text(
        "---\npriority: 0\nworking_directory: .\nmax_retries: 3\n---\n\nDaily review prompt"
    )
    prompt = storage.use_bank_template("daily")
    assert prompt is not None
    state = QueueState()
    state.add_prompt(prompt)
    storage.save_queue_state(state)
    files = list(storage.queue_dir.glob("*.md"))
    assert len(files) == 1


def test_bank_use_preserves_priority_from_template(tmp_path):  # STO-058
    """Priority from the bank template is copied to the new QueuedPrompt."""
    storage = QueueStorage(str(tmp_path))
    storage.save_prompt_to_bank("priority-test", priority=2)
    (storage.bank_dir / "priority-test.md").write_text(
        "---\npriority: 2\nworking_directory: .\nmax_retries: 3\n---\n\nContent"
    )
    prompt = storage.use_bank_template("priority-test")
    assert prompt is not None
    assert prompt.priority == 2


def test_bank_use_preserves_max_retries_from_template(tmp_path):  # STO-059
    """max_retries from the bank template is copied to the new QueuedPrompt."""
    storage = QueueStorage(str(tmp_path))
    (storage.bank_dir / "retry-test.md").write_text(
        "---\npriority: 0\nworking_directory: .\nmax_retries: 5\n---\n\nContent"
    )
    prompt = storage.use_bank_template("retry-test")
    assert prompt is not None
    assert prompt.max_retries == 5


def test_bank_delete_removes_template(tmp_path):  # STO-060
    """delete_bank_template() removes the bank file."""
    storage = QueueStorage(str(tmp_path))
    storage.save_prompt_to_bank("temp")
    storage.delete_bank_template("temp")
    assert not (storage.bank_dir / "temp.md").exists()


def test_bank_delete_nonexistent_returns_false(tmp_path):  # STO-061
    """delete_bank_template() returns False when the template does not exist."""
    storage = QueueStorage(str(tmp_path))
    result = storage.delete_bank_template("does-not-exist")
    assert result is False


def test_bank_template_priority_configurable(tmp_path):  # STO-062
    """save_prompt_to_bank('task', priority=5) writes 'priority: 5' in the YAML."""
    storage = QueueStorage(str(tmp_path))
    storage.save_prompt_to_bank("my-task", priority=5)
    content = (storage.bank_dir / "my-task.md").read_text()
    assert "priority: 5" in content


def test_bank_use_nonexistent_template_returns_none(tmp_path):  # STO-063
    """use_bank_template() returns None when the template does not exist."""
    storage = QueueStorage(str(tmp_path))
    result = storage.use_bank_template("nonexistent")
    assert result is None


def test_bank_list_returns_full_metadata(tmp_path):  # STO-064
    """list_bank_templates() dicts include name, title, priority, working_directory,
    estimated_tokens (None by default), and a non-None modified timestamp.
    """
    storage = QueueStorage(str(tmp_path))
    storage.save_prompt_to_bank("my-template", priority=3)
    templates = storage.list_bank_templates()
    assert len(templates) == 1
    t = templates[0]
    assert t["name"] == "my-template"
    assert t["priority"] == 3
    assert t["working_directory"] == "."
    assert t["estimated_tokens"] is None
    assert "modified" in t and t["modified"] is not None
    assert "title" in t and len(t["title"]) > 0

"""
Fault tolerance tests — file system errors, subprocess death, crash recovery,
network failures, authentication failures, and complex failure scenarios.

Label scheme: FT-001 through FT-085.

Sections:
  FT-001..013  File system write errors         (class TestWriteErrors)
  FT-014..020  File system read errors          (class TestReadErrors)
  FT-021..023  Disk-full (ENOSPC)               (class TestDiskFull)
  FT-024..028  TOCTOU / deletion races          (class TestTOCTOU)
  FT-029..030  Init failures                    (class TestInitErrors)
  FT-031..036  Manager storage errors           (class TestManagerErrors)
  FT-037..043  Known bugs (xfail)               (class TestKnownBugs)
  FT-044..056  Subprocess death                 (module-level)
  FT-057..066  SIGKILL / crash recovery         (class TestSIGKILLRecovery)
  FT-067..074  Network failures                 (module-level)
  FT-075..083  Authentication failures          (module-level)
  FT-084..085  Complex failure scenarios        (module-level)
"""

import builtins
import errno
import os
import subprocess
from pathlib import Path
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
from claude_code_queue.storage import MarkdownPromptParser, QueueStorage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ENOSPC_ERROR = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

VALID_FRONTMATTER = (
    "---\n"
    "priority: 0\n"
    "working_directory: .\n"
    "max_retries: 3\n"
    "status: queued\n"
    "retry_count: 0\n"
    "created_at: 2025-01-01T00:00:00\n"
    "---\n\n"
    "task"
)

_EXEC_FRONTMATTER = (
    "---\n"
    "priority: 0\n"
    "working_directory: .\n"
    "max_retries: 3\n"
    "status: executing\n"
    "retry_count: {retry_count}\n"
    "created_at: 2025-01-01T00:00:00\n"
    "---\n\n"
    "{content}"
)


def _write_executing_file(queue_dir, prompt_id, slug, content="task", retry_count=0):
    path = queue_dir / f"{prompt_id}-{slug}.executing.md"
    path.write_text(
        _EXEC_FRONTMATTER.format(retry_count=retry_count, content=content)
    )
    return path


def _fail_result(error: str = "exit 1") -> ExecutionResult:
    return ExecutionResult(
        success=False,
        output="",
        error=error,
        execution_time=0.1,
        rate_limit_info=RateLimitInfo(is_rate_limited=False),
    )


def _timeout_result() -> ExecutionResult:
    return ExecutionResult(
        success=False,
        output="",
        error="Execution timed out after 3600 seconds",
        execution_time=3600.1,
        rate_limit_info=RateLimitInfo(is_rate_limited=False),
    )


def _network_fail_result() -> ExecutionResult:
    return ExecutionResult(
        success=False,
        output="",
        error="curl: (6) Could not resolve host: api.anthropic.com",
        execution_time=30.0,
        rate_limit_info=RateLimitInfo(is_rate_limited=False),
    )


def _auth_fail_result(error: str = "Error: Not authenticated. Please run 'claude login'.") -> ExecutionResult:
    return ExecutionResult(
        success=False,
        output="",
        error=error,
        execution_time=0.3,
        rate_limit_info=RateLimitInfo(is_rate_limited=False),
    )


def _success_result(output: str = "done") -> ExecutionResult:
    return ExecutionResult(
        success=True,
        output=output,
        error="",
        execution_time=1.0,
        rate_limit_info=RateLimitInfo(is_rate_limited=False),
    )


# ---------------------------------------------------------------------------
# Module-level fixtures (used by subprocess death, network, auth, scenario tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def iface():
    """ClaudeCodeInterface with __init__ bypassed via __new__."""
    obj = ClaudeCodeInterface.__new__(ClaudeCodeInterface)
    obj.claude_command = "claude"
    obj.timeout = 3600
    obj.skip_permissions = True
    return obj


@pytest.fixture
def mock_iface():
    """Mocked ClaudeCodeInterface."""
    m = MagicMock(spec=ClaudeCodeInterface)
    m.test_connection.return_value = (True, "OK")
    return m


@pytest.fixture
def mgr(tmp_path, mock_iface):
    """QueueManager with real storage and mocked claude interface (running=False)."""
    m = QueueManager.__new__(QueueManager)
    m.storage = QueueStorage(str(tmp_path))
    m.claude_interface = mock_iface
    m.check_interval = 30
    m.running = False
    m.state = None
    return m


# ===========================================================================
# File System Write Errors
# ===========================================================================


class TestWriteErrors:
    """FT-001..013: write-permission / write-error failures."""

    def test_write_prompt_file_returns_false_on_permission_error(self, tmp_path):  # FT-001
        """write_prompt_file returns False when open() raises PermissionError."""
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task")
        file_path = storage.queue_dir / "abc12345-task.md"

        with patch("builtins.open", side_effect=PermissionError("read-only")):
            result = storage.parser.write_prompt_file(prompt, file_path)

        assert result is False

    def test_save_queue_state_returns_false_on_json_write_error(self, tmp_path):  # FT-002
        """save_queue_state returns False when the state JSON cannot be opened."""
        storage = QueueStorage(str(tmp_path))
        state = QueueState()

        real_open = builtins.open

        def selective_open(path, *args, **kwargs):
            if "queue-state.json" in str(path) and args and "w" in args[0]:
                raise PermissionError("read-only")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_open):
            result = storage.save_queue_state(state)

        assert result is False

    def test_in_memory_state_unchanged_when_save_fails(self, tmp_path):  # FT-003
        """save_queue_state never mutates the state object it receives."""
        storage = QueueStorage(str(tmp_path))
        state = QueueState()
        state.total_processed = 7
        state.failed_count = 2
        prompt = QueuedPrompt(id="abc12345", content="task")
        state.add_prompt(prompt)

        with patch("builtins.open", side_effect=PermissionError("read-only")):
            storage.save_queue_state(state)

        assert state.total_processed == 7
        assert state.failed_count == 2
        assert len(state.prompts) == 1
        assert state.prompts[0].id == "abc12345"

    def test_save_single_prompt_catches_unexpected_write_exception(self, tmp_path):  # FT-004
        """_save_single_prompt outer except catches RuntimeError from write_prompt_file."""
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task")

        with patch.object(
            storage.parser,
            "write_prompt_file",
            side_effect=RuntimeError("unexpected internal error"),
        ):
            result = storage._save_single_prompt(prompt)

        assert result is False

    def test_save_single_prompt_propagates_false_from_write_prompt_file(self, tmp_path):  # FT-005
        """_save_single_prompt returns False when write_prompt_file returns False."""
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task")

        with patch.object(storage.parser, "write_prompt_file", return_value=False):
            result = storage._save_single_prompt(prompt)

        assert result is False

    def test_executing_save_failure_loses_queued_file(self, tmp_path):  # FT-006
        """When EXECUTING write fails, _remove_prompt_files has already deleted the
        QUEUED file — the prompt vanishes from all directories (documents data-loss bug).
        """
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(
            id="abc12345", content="task", status=PromptStatus.EXECUTING
        )

        queue_file = storage.queue_dir / "abc12345-task.md"
        queue_file.write_text(VALID_FRONTMATTER)

        with patch.object(storage.parser, "write_prompt_file", return_value=False):
            storage._save_single_prompt(prompt)

        assert not queue_file.exists(), (
            "QUEUED file deleted even when EXECUTING write failed — data loss"
        )
        assert list(storage.queue_dir.glob("abc12345*.executing.md")) == [], (
            "No EXECUTING file exists after write failure"
        )

    def test_completed_prompt_not_lost_when_write_fails(self, tmp_path):  # FT-007
        """When COMPLETED write fails the queue file must survive (write-then-delete order).
        This assertion FAILS against current code — documents the delete-before-write bug.
        """
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task", status=PromptStatus.QUEUED)
        queue_file = storage.queue_dir / "abc12345-task.md"
        queue_file.write_text(VALID_FRONTMATTER)

        prompt.status = PromptStatus.COMPLETED
        with patch.object(storage.parser, "write_prompt_file", return_value=False):
            storage._save_single_prompt(prompt)

        surviving = (
            list(storage.queue_dir.glob("abc12345*.md"))
            + list(storage.completed_dir.glob("abc12345*.md"))
        )
        assert len(surviving) > 0, (
            "Prompt must not vanish from all directories when completed write fails"
        )

    @pytest.mark.parametrize(
        "exc",
        [
            PermissionError("read-only"),
            OSError(errno.ENOSPC, os.strerror(errno.ENOSPC)),
        ],
        ids=["permission_error", "enospc"],
    )
    def test_create_prompt_template_raises_on_write_error(self, tmp_path, exc):  # FT-008
        """create_prompt_template has no try/except — write errors propagate."""
        storage = QueueStorage(str(tmp_path))
        with patch("builtins.open", side_effect=exc):
            with pytest.raises(OSError):
                storage.create_prompt_template("my-task")

    @pytest.mark.parametrize(
        "exc",
        [
            PermissionError("read-only"),
            OSError(errno.ENOSPC, os.strerror(errno.ENOSPC)),
        ],
        ids=["permission_error", "enospc"],
    )
    def test_save_prompt_to_bank_raises_on_write_error(self, tmp_path, exc):  # FT-009
        """save_prompt_to_bank has no try/except — write errors propagate."""
        storage = QueueStorage(str(tmp_path))
        with patch("builtins.open", side_effect=exc):
            with pytest.raises(OSError):
                storage.save_prompt_to_bank("my-template")

    def test_delete_bank_template_returns_false_when_unlink_fails(self, tmp_path):  # FT-010
        """delete_bank_template catches unlink errors and returns False."""
        storage = QueueStorage(str(tmp_path))
        storage.save_prompt_to_bank("temp")

        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            result = storage.delete_bank_template("temp")

        assert result is False

    def test_remove_prompt_files_does_not_raise_when_unlink_fails(self, tmp_path):  # FT-011
        """_remove_prompt_files swallows unlink errors and returns normally."""
        storage = QueueStorage(str(tmp_path))
        prompt_id = "abc12345"
        (storage.queue_dir / f"{prompt_id}-task.md").write_text("content")
        (storage.queue_dir / f"{prompt_id}-task.executing.md").write_text("content")

        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            storage._remove_prompt_files(prompt_id, storage.queue_dir)

    def test_write_prompt_file_returns_false_when_yaml_dump_raises(self, tmp_path):  # FT-012
        """If yaml.dump raises mid-write, write_prompt_file must return False."""
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task")
        prompt.add_log("Step 1 done")
        file_path = storage.queue_dir / "abc12345-task.md"

        with patch(
            "claude_code_queue.storage.yaml.dump",
            side_effect=OSError(errno.ENOSPC, "No space left on device"),
        ):
            result = storage.parser.write_prompt_file(prompt, file_path)

        assert result is False

    def test_add_prompt_from_markdown_raises_on_move_error(self, tmp_path):  # FT-013
        """add_prompt_from_markdown propagates shutil.move errors unchecked."""
        storage = QueueStorage(str(tmp_path))
        external_file = tmp_path / "my-prompt.md"
        external_file.write_text(
            "---\npriority: 0\nworking_directory: .\nmax_retries: 3\n---\n\nDo something."
        )

        with patch(
            "claude_code_queue.storage.shutil.move",
            side_effect=OSError("cross-device link"),
        ):
            with pytest.raises(OSError):
                storage.add_prompt_from_markdown(external_file)


# ===========================================================================
# File System Read Errors
# ===========================================================================


class TestReadErrors:
    """FT-014..020: read-permission / read-error failures."""

    def test_parse_prompt_file_returns_none_on_permission_error(self, tmp_path):  # FT-014
        """parse_prompt_file returns None when open() raises PermissionError."""
        storage = QueueStorage(str(tmp_path))
        file_path = storage.queue_dir / "abc12345-task.md"
        file_path.write_text("---\n---\n\ntask")

        with patch("builtins.open", side_effect=PermissionError("no permission")):
            result = storage.parser.parse_prompt_file(file_path)

        assert result is None

    def test_load_queue_state_skips_unreadable_prompt_file(self, tmp_path):  # FT-015
        """One unreadable prompt file is skipped; others load correctly."""
        storage = QueueStorage(str(tmp_path))
        (storage.queue_dir / "abc12345-task-one.md").write_text(VALID_FRONTMATTER)
        (storage.queue_dir / "def67890-task-two.md").write_text(
            VALID_FRONTMATTER.replace("task", "task two")
        )

        real_open = builtins.open

        def selective_open(path, *args, **kwargs):
            if "abc12345" in str(path):
                raise PermissionError("no permission")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_open):
            state = storage.load_queue_state()

        loaded_ids = {p.id for p in state.prompts}
        assert "abc12345" not in loaded_ids
        assert "def67890" in loaded_ids
        assert len(state.prompts) == 1

    def test_load_queue_state_raises_when_load_prompts_raises(self, tmp_path):  # FT-016
        """load_queue_state propagates exceptions from _load_prompts_from_files."""
        storage = QueueStorage(str(tmp_path))

        with patch.object(
            storage,
            "_load_prompts_from_files",
            side_effect=PermissionError("queue dir not readable"),
        ):
            with pytest.raises(PermissionError):
                storage.load_queue_state()

    def test_load_queue_state_falls_back_when_state_json_unreadable(self, tmp_path):  # FT-017
        """Unreadable queue-state.json causes counters to fall back to zero."""
        storage = QueueStorage(str(tmp_path))
        state = QueueState()
        state.total_processed = 10
        storage.save_queue_state(state)
        assert storage.state_file.exists()

        real_open = builtins.open

        def selective_open(path, *args, **kwargs):
            if "queue-state.json" in str(path) and args and "r" in args[0]:
                raise PermissionError("no permission")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_open):
            loaded = storage.load_queue_state()

        assert loaded.total_processed == 0
        assert loaded.failed_count == 0

    def test_list_bank_templates_skips_unreadable_files(self, tmp_path):  # FT-018
        """list_bank_templates skips unreadable files and returns the rest."""
        storage = QueueStorage(str(tmp_path))
        storage.save_prompt_to_bank("alpha")
        storage.save_prompt_to_bank("beta")

        real_open = builtins.open

        def selective_open(path, *args, **kwargs):
            if "alpha" in str(path):
                raise PermissionError("no permission")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_open):
            templates = storage.list_bank_templates()

        assert len(templates) == 1
        assert templates[0]["name"] == "beta"

    def test_use_bank_template_returns_none_when_file_unreadable(self, tmp_path):  # FT-019
        """use_bank_template returns None when the bank file exists but is unreadable."""
        storage = QueueStorage(str(tmp_path))
        storage.save_prompt_to_bank("daily")

        with patch("builtins.open", side_effect=PermissionError("no permission")):
            result = storage.use_bank_template("daily")

        assert result is None

    def test_add_prompt_from_markdown_returns_none_when_unreadable(self, tmp_path):  # FT-020
        """add_prompt_from_markdown returns None when the source file is unreadable."""
        storage = QueueStorage(str(tmp_path))
        external_file = tmp_path / "my-prompt.md"
        external_file.write_text("---\n---\n\nDo something.")

        with patch("builtins.open", side_effect=PermissionError("no permission")):
            result = storage.add_prompt_from_markdown(external_file)

        assert result is None
        assert external_file.exists()


# ===========================================================================
# Disk-Full (ENOSPC) Failures
# ===========================================================================


class TestDiskFull:
    """FT-021..023: ENOSPC failures (unique cases)."""

    def test_write_prompt_file_returns_false_on_enospc(self, tmp_path):  # FT-021
        """write_prompt_file returns False on ENOSPC from open()."""
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task")
        file_path = storage.queue_dir / "abc12345-task.md"

        with patch(
            "builtins.open",
            side_effect=OSError(errno.ENOSPC, os.strerror(errno.ENOSPC)),
        ):
            result = storage.parser.write_prompt_file(prompt, file_path)

        assert result is False

    def test_save_queue_state_returns_true_despite_prompt_write_failures(self, tmp_path):  # FT-022
        """Silent partial-failure bug: save_queue_state returns True even when every
        prompt write failed (documents the false-success bug).
        """
        storage = QueueStorage(str(tmp_path))
        state = QueueState()
        state.add_prompt(QueuedPrompt(id="abc12345", content="task"))

        with patch.object(storage.parser, "write_prompt_file", return_value=False):
            result = storage.save_queue_state(state)

        assert result is True
        assert list(storage.queue_dir.glob("*.md")) == []

    def test_save_queue_state_returns_false_on_enospc_for_state_json(self, tmp_path):  # FT-023
        """save_queue_state returns False when queue-state.json write hits ENOSPC."""
        storage = QueueStorage(str(tmp_path))
        state = QueueState()

        real_open = builtins.open

        def selective_open(path, *args, **kwargs):
            if "queue-state.json" in str(path) and args and "w" in args[0]:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_open):
            result = storage.save_queue_state(state)

        assert result is False


# ===========================================================================
# TOCTOU / Deletion-Between-Operations
# ===========================================================================


class TestTOCTOU:
    """FT-024..028: time-of-check/time-of-use races."""

    def test_file_deleted_between_glob_and_parse_does_not_crash(self, tmp_path):  # FT-024
        """File deleted between glob and parse is silently skipped."""
        storage = QueueStorage(str(tmp_path))
        (storage.queue_dir / "abc12345-task.md").write_text(VALID_FRONTMATTER)

        original_parse = storage.parser.__class__.parse_prompt_file

        def delete_then_parse(path):
            path.unlink(missing_ok=True)
            return original_parse(path)

        with patch.object(
            storage.parser, "parse_prompt_file", side_effect=delete_then_parse
        ):
            state = storage.load_queue_state()

        assert len(state.prompts) == 0

    def test_remove_prompt_files_handles_file_not_found_from_unlink(self, tmp_path):  # FT-025
        """_remove_prompt_files swallows FileNotFoundError from unlink."""
        storage = QueueStorage(str(tmp_path))
        (storage.queue_dir / "abc12345-task.md").write_text("content")

        with patch.object(
            Path, "unlink", side_effect=FileNotFoundError("already deleted")
        ):
            storage._remove_prompt_files("abc12345", storage.queue_dir)

    @pytest.mark.parametrize(
        "status",
        [
            PromptStatus.COMPLETED,
            PromptStatus.FAILED,
            PromptStatus.CANCELLED,
        ],
    )
    def test_save_single_prompt_write_failure_cross_dir_preserves_queue_file(
        self, tmp_path, status
    ):  # FT-026 (updated for FT-007 fix)
        """Write-then-delete for cross-directory transitions: if the destination
        write fails, the source queue file must survive.  FT-007 fix changed the
        order from delete-before-write to write-then-conditional-delete.
        """
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task", status=PromptStatus.QUEUED)
        queue_file = storage.queue_dir / "abc12345-task.md"
        queue_file.write_text(VALID_FRONTMATTER)

        prompt.status = status
        with patch.object(storage.parser, "write_prompt_file", return_value=False):
            storage._save_single_prompt(prompt)

        surviving_queue_files = list(storage.queue_dir.glob("abc12345-*.md"))
        assert len(surviving_queue_files) == 1, (
            f"Queue file lost on write failure for status={status.value} "
            f"(FT-007: write-then-delete must preserve source on write failure)"
        )

    def test_save_single_prompt_executing_write_failure_loses_file(self, tmp_path):  # FT-026b
        """Same-directory transition (EXECUTING) still uses delete-before-write:
        a write failure leaves the prompt in no directory.  This is an accepted
        trade-off; crash-recovery via load_queue_state handles it.
        """
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task", status=PromptStatus.QUEUED)
        queue_file = storage.queue_dir / "abc12345-task.md"
        queue_file.write_text(VALID_FRONTMATTER)

        prompt.status = PromptStatus.EXECUTING
        with patch.object(storage.parser, "write_prompt_file", return_value=False):
            storage._save_single_prompt(prompt)

        all_prompt_files = (
            list(storage.queue_dir.glob("abc12345-*.md"))
            + list(storage.completed_dir.glob("abc12345-*.md"))
            + list(storage.failed_dir.glob("abc12345-*.md"))
        )
        assert len(all_prompt_files) == 0, (
            "EXECUTING same-dir transition: file expected to be gone after write failure"
        )

    def test_stat_failure_in_parse_prompt_file_returns_none(self, tmp_path):  # FT-027
        """parse_prompt_file returns None when stat() raises after a successful open.

        Uses a file without created_at so the fallback stat() path is exercised.
        """
        storage = QueueStorage(str(tmp_path))
        file_path = storage.queue_dir / "abc12345-task.md"
        file_path.write_text(
            "---\n"
            "priority: 0\n"
            "working_directory: .\n"
            "max_retries: 3\n"
            "status: queued\n"
            "retry_count: 0\n"
            "---\n\n"
            "task"
        )

        with patch.object(
            Path, "stat", side_effect=FileNotFoundError("file gone after open")
        ):
            result = storage.parser.parse_prompt_file(file_path)

        assert result is None

    def test_list_bank_templates_stat_failure_skips_file(self, tmp_path):  # FT-028
        """list_bank_templates catches per-file stat() errors and continues."""
        storage = QueueStorage(str(tmp_path))
        storage.save_prompt_to_bank("alpha")
        storage.save_prompt_to_bank("beta")

        original_stat = Path.stat

        def stat_side_effect(self_path):
            if self_path.name == "alpha.md":
                raise FileNotFoundError("file gone")
            return original_stat(self_path)

        with patch.object(Path, "stat", stat_side_effect):
            templates = storage.list_bank_templates()

        assert len(templates) == 1


# ===========================================================================
# Init Failures
# ===========================================================================


class TestInitErrors:
    """FT-029..030: QueueStorage initialisation failures."""

    @pytest.mark.parametrize(
        "exc",
        [
            PermissionError("parent dir not writable"),
            OSError(errno.ENOSPC, os.strerror(errno.ENOSPC)),
        ],
        ids=["permission_error", "enospc"],
    )
    def test_queue_storage_init_propagates_mkdir_error(self, tmp_path, exc):  # FT-029
        """QueueStorage.__init__ propagates mkdir errors (no try/except)."""
        with patch.object(Path, "mkdir", side_effect=exc):
            with pytest.raises(OSError):
                QueueStorage(str(tmp_path))

    def test_queue_storage_init_succeeds_when_dirs_already_exist(self, tmp_path):  # FT-030
        """Constructing QueueStorage twice on the same directory never raises."""
        QueueStorage(str(tmp_path))
        QueueStorage(str(tmp_path))


# ===========================================================================
# Manager-Level Behaviour Under Storage Errors
# ===========================================================================


class TestManagerErrors:
    """FT-031..036: QueueManager behaviour when storage operations fail."""

    @pytest.fixture()
    def manager(self, tmp_path, mocker):
        mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
        mocker.patch.object(
            ClaudeCodeInterface, "test_connection", return_value=(True, "ok")
        )
        return QueueManager(storage_dir=str(tmp_path), claude_command="claude")

    def test_add_prompt_returns_false_on_storage_failure(self, manager):  # FT-031
        """add_prompt returns False when save_queue_state returns False."""
        p = QueuedPrompt(content="new task")
        manager.state = manager.storage.load_queue_state()

        with patch.object(manager.storage, "save_queue_state", return_value=False):
            result = manager.add_prompt(p)

        assert result is False

    def test_remove_prompt_returns_false_on_storage_failure(self, manager):  # FT-032
        """remove_prompt returns False when save_queue_state returns False."""
        p = QueuedPrompt(content="task")
        manager.state = manager.storage.load_queue_state()
        manager.state.add_prompt(p)

        with patch.object(manager.storage, "save_queue_state", return_value=False):
            result = manager.remove_prompt(p.id)

        assert result is False

    def test_shutdown_raises_when_save_queue_state_raises(self, manager):  # FT-033
        """_shutdown propagates exceptions from save_queue_state (no try/except)."""
        manager.state = QueueState()

        with patch.object(
            manager.storage,
            "save_queue_state",
            side_effect=Exception("disk error"),
        ):
            with pytest.raises(Exception, match="disk error"):
                manager._shutdown()

    def test_execute_prompt_raises_when_pre_execution_save_raises(self, manager):  # FT-034
        """_execute_prompt propagates exception from pre-execution save_queue_state."""
        manager.state = QueueState()
        prompt = QueuedPrompt(content="task", status=PromptStatus.EXECUTING)
        manager.state.add_prompt(prompt)

        with patch.object(
            manager.storage,
            "save_queue_state",
            side_effect=Exception("disk write failure"),
        ):
            with pytest.raises(Exception, match="disk write failure"):
                manager._execute_prompt(prompt)

    def test_process_queue_iteration_raises_when_post_execution_save_raises(self, manager):  # FT-035
        """_process_queue_iteration propagates exception from post-execution save."""
        state = manager.storage.load_queue_state()
        p = QueuedPrompt(content="task")
        state.add_prompt(p)
        manager.storage.save_queue_state(state)

        call_count = 0
        real_save = manager.storage.save_queue_state

        def save_side_effect(s):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("disk full")
            return real_save(s)

        with patch.object(
            manager.claude_interface,
            "execute_prompt",
            return_value=ExecutionResult(
                success=True, output="done", execution_time=0.1
            ),
        ):
            with patch.object(
                manager.storage, "save_queue_state", side_effect=save_side_effect
            ):
                manager.state = None
                with pytest.raises(Exception, match="disk full"):
                    manager._process_queue_iteration()

    def test_process_queue_iteration_raises_when_load_raises(self, manager):  # FT-036
        """_process_queue_iteration propagates exception from load_queue_state."""
        with patch.object(
            manager.storage,
            "load_queue_state",
            side_effect=Exception("disk unreadable"),
        ):
            with pytest.raises(Exception, match="disk unreadable"):
                manager._process_queue_iteration()


# ===========================================================================
# Known Bugs (xfail)
# ===========================================================================


class TestKnownBugs:
    """FT-037..043: real bugs discovered during test plan analysis."""

    def test_remove_prompt_files_does_not_delete_files_with_prefix_match(self, tmp_path):  # FT-037
        """BUG: glob pattern '{id}*.md' matches IDs that start with the given id.
        Removing 'abc1' must NOT delete 'abc12345-task.md'.
        """
        storage = QueueStorage(str(tmp_path))
        (storage.queue_dir / "abc1-task.md").write_text("content for abc1")
        (storage.queue_dir / "abc12345-task.md").write_text("content for abc12345")

        storage._remove_prompt_files("abc1", storage.queue_dir)

        assert not (storage.queue_dir / "abc1-task.md").exists()
        assert (storage.queue_dir / "abc12345-task.md").exists(), (
            "BUG: prefix glob deleted file for a different prompt (abc12345 != abc1)"
        )

    def test_list_bank_templates_with_file_without_frontmatter(self, tmp_path):  # FT-038
        """list_bank_templates must not skip a bank file that has no frontmatter.
        Correct behavior: return the template with the filename as its title.
        """
        storage = QueueStorage(str(tmp_path))
        (storage.bank_dir / "plain.md").write_text(
            "Just plain text.\nNo frontmatter here."
        )

        templates = storage.list_bank_templates()

        assert len(templates) == 1, (
            "BUG: bank file without frontmatter silently skipped due to NameError in 'parts'"
        )
        assert templates[0]["name"] == "plain"

    def test_write_prompt_file_leaves_partial_file_on_enospc_during_yaml_dump(self, tmp_path):  # FT-039
        """write_prompt_file creates/truncates the file before yaml.dump is called.
        If yaml.dump raises ENOSPC, a partial (invalid) file remains on disk.
        """
        storage = QueueStorage(str(tmp_path))
        prompt = QueuedPrompt(id="abc12345", content="task")
        file_path = storage.queue_dir / "abc12345-task.md"

        with patch(
            "claude_code_queue.storage.yaml.dump",
            side_effect=OSError(errno.ENOSPC, os.strerror(errno.ENOSPC)),
        ):
            result = storage.parser.write_prompt_file(prompt, file_path)

        assert result is False
        assert file_path.exists(), "Partial file should exist after mid-write failure"
        content = file_path.read_text()
        assert content == "---\n", (
            f"Expected only partial write '---\\n', got: {content!r}"
        )

    def test_save_queue_state_leaves_partial_json_on_enospc_during_json_dump(self, tmp_path):  # FT-040
        """save_queue_state truncates queue-state.json before json.dump. If json.dump
        raises ENOSPC the state file is empty — counter history is lost on next load.
        """
        storage = QueueStorage(str(tmp_path))
        state = QueueState()
        state.total_processed = 42
        storage.save_queue_state(state)

        with patch(
            "claude_code_queue.storage.json.dump",
            side_effect=OSError(errno.ENOSPC, os.strerror(errno.ENOSPC)),
        ):
            result = storage.save_queue_state(state)

        assert result is False

        content = storage.state_file.read_text()
        assert len(content) == 0, (
            "State file must be empty (truncated before json.dump failed)"
        )

        reloaded = storage.load_queue_state()
        assert reloaded.total_processed == 0, (
            "Counter history lost due to partial write"
        )

    def test_add_prompt_creates_memory_disk_inconsistency_on_save_failure(self, tmp_path, mocker):  # FT-041
        """add_prompt adds the prompt to in-memory state BEFORE calling save_queue_state.
        If save fails, memory and disk diverge.
        """
        mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
        mocker.patch.object(
            ClaudeCodeInterface, "test_connection", return_value=(True, "ok")
        )
        manager = QueueManager(storage_dir=str(tmp_path), claude_command="claude")

        p = QueuedPrompt(content="task")
        manager.state = manager.storage.load_queue_state()

        with patch.object(manager.storage, "save_queue_state", return_value=False):
            result = manager.add_prompt(p)

        assert result is False
        assert any(pr.id == p.id for pr in manager.state.prompts), (
            "Prompt was added to in-memory state even though disk save failed"
        )

        reloaded = manager.storage.load_queue_state()
        assert not any(pr.id == p.id for pr in reloaded.prompts), (
            "Prompt correctly absent from disk state (save failed)"
        )

    def test_create_prompt_template_path_traversal_blocked(self, tmp_path):  # FT-042
        """BUG: create_prompt_template does not sanitize filenames. A path-traversal
        name like '../../malicious' must not write outside queue_dir.
        """
        storage = QueueStorage(str(tmp_path))
        evil_name = "../../malicious"

        try:
            storage.create_prompt_template(evil_name)
        except Exception:
            pass

        evil_path = tmp_path.parent / "malicious.md"
        assert not evil_path.exists(), (
            "BUG: path traversal via template name allows writing outside queue_dir"
        )

    def test_save_prompt_to_bank_path_traversal_blocked(self, tmp_path):  # FT-043
        """BUG: save_prompt_to_bank does not sanitize template_name. A path-traversal
        name like '../../malicious' must not write outside bank_dir.
        """
        storage = QueueStorage(str(tmp_path))
        evil_name = "../../malicious"

        try:
            storage.save_prompt_to_bank(evil_name)
        except Exception:
            pass

        evil_path = tmp_path.parent / "malicious.md"
        assert not evil_path.exists(), (
            "BUG: path traversal via template_name allows writing outside bank_dir"
        )


# ===========================================================================
# Subprocess Death (T01–T13)
# ===========================================================================


def test_subprocess_nonzero_returncode_yields_failure_result(iface, mocker, tmp_path):  # FT-044
    """returncode=137 (SIGKILL) → success=False and is_rate_limited=False."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.return_value = MagicMock(returncode=137, stdout="", stderr="Killed")

    prompt = QueuedPrompt(
        id="t01aaaa", content="fix a bug", working_directory=str(tmp_path)
    )
    result = iface.execute_prompt(prompt)

    assert result.success is False
    assert result.is_rate_limited is False
    assert "Killed" in result.error or result.error == ""


def test_subprocess_timeout_returns_timeout_failure(iface, mocker, tmp_path):  # FT-045
    """subprocess.TimeoutExpired is caught and returned as a failure, not re-raised."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=10)

    prompt = QueuedPrompt(
        id="t02aaaa", content="some task", working_directory=str(tmp_path)
    )
    result = iface.execute_prompt(prompt)

    assert result.success is False
    assert result.is_rate_limited is False
    assert "timed out" in result.error.lower()
    assert result.execution_time >= 0


def test_subprocess_oserror_returns_failure_not_exception(iface, mocker, tmp_path):  # FT-046
    """OSError inside execute_prompt() is caught and returned as failure."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.side_effect = OSError("Connection reset by peer")

    prompt = QueuedPrompt(
        id="t03aaaa", content="some task", working_directory=str(tmp_path)
    )
    result = iface.execute_prompt(prompt)

    assert result.success is False
    assert "Connection reset by peer" in result.error


def test_subprocess_generic_exception_returns_failure(iface, mocker, tmp_path):  # FT-047
    """RuntimeError inside execute_prompt() is caught via generic except clause."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.side_effect = RuntimeError("unexpected internal error")

    prompt = QueuedPrompt(
        id="t04aaaa", content="task", working_directory=str(tmp_path)
    )
    result = iface.execute_prompt(prompt)

    assert result.success is False
    assert "unexpected internal error" in result.error


def test_working_dir_restored_after_subprocess_timeout(iface, mocker, tmp_path):  # FT-048
    """os.getcwd() is unchanged after TimeoutExpired (cwd= param, not os.chdir)."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    original_cwd = os.getcwd()

    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=10)

    prompt = QueuedPrompt(
        id="t05aaaa", content="task", working_directory=str(work_dir)
    )
    iface.execute_prompt(prompt)

    assert os.getcwd() == original_cwd


def test_working_dir_restored_after_generic_exception(iface, mocker, tmp_path):  # FT-049
    """os.getcwd() is unchanged after RuntimeError (cwd= param, not os.chdir)."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    original_cwd = os.getcwd()

    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.side_effect = RuntimeError("boom")

    prompt = QueuedPrompt(
        id="t06aaaa", content="task", working_directory=str(work_dir)
    )
    iface.execute_prompt(prompt)

    assert os.getcwd() == original_cwd


def test_execute_prompt_returncode_137_not_rate_limited(iface, mocker, tmp_path):  # FT-050
    """SIGKILL exit (returncode 137) with 'Killed' in stdout is NOT rate-limited."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.return_value = MagicMock(returncode=137, stdout="Killed", stderr="")

    prompt = QueuedPrompt(
        id="t07aaaa", content="some work", working_directory=str(tmp_path)
    )
    result = iface.execute_prompt(prompt)

    assert result.is_rate_limited is False


def test_detect_rate_limit_not_triggered_by_empty_output(iface):  # FT-051
    """Empty string passed to _detect_rate_limit() returns is_rate_limited=False."""
    info = iface._detect_rate_limit("")
    assert info.is_rate_limited is False


def test_failed_subprocess_with_retries_remaining_queues_retry(mgr, mock_iface, tmp_path):  # FT-052
    """After a subprocess failure with retries remaining, prompt goes back to QUEUED."""
    mock_iface.execute_prompt.return_value = _fail_result("exit 1")

    prompt = QueuedPrompt(
        id="t09aaaa", content="do something", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)
    mgr.state = None

    mgr._process_queue_iteration()

    assert mgr.state.prompts[0].status == PromptStatus.QUEUED
    assert mgr.state.prompts[0].retry_count == 1
    assert mgr.state.failed_count == 0


def test_failed_subprocess_exhausting_retries_marks_failed(mgr, mock_iface):  # FT-053
    """When retry_count == max_retries before execution, next failure is permanent FAILED."""
    mock_iface.execute_prompt.return_value = _fail_result("exit 1")

    prompt = QueuedPrompt(
        id="t10aaaa", content="do something", max_retries=3, retry_count=3
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)
    mgr.state = None

    mgr._process_queue_iteration()

    assert mgr.state.prompts[0].status == PromptStatus.FAILED
    assert mgr.state.failed_count == 1


def test_timeout_failure_with_retries_remaining_queues_retry(mgr, mock_iface):  # FT-054
    """A timeout ExecutionResult with retries remaining puts the prompt back to QUEUED."""
    mock_iface.execute_prompt.return_value = _timeout_result()

    prompt = QueuedPrompt(
        id="t11aaaa", content="long task", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)
    mgr.state = None

    mgr._process_queue_iteration()

    assert mgr.state.prompts[0].status == PromptStatus.QUEUED
    assert mgr.state.prompts[0].retry_count == 1


def test_repeated_subprocess_kills_exhaust_retries_and_fail(mgr, mock_iface):  # FT-055
    """Subprocess killed on every attempt: after max_retries the prompt is FAILED."""
    mock_iface.execute_prompt.return_value = _fail_result("killed")

    prompt = QueuedPrompt(
        id="t12aaaa", content="work", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)

    for _ in range(3):
        mgr.state = None
        mgr._process_queue_iteration()

    assert mgr.state.prompts[0].status == PromptStatus.FAILED
    assert mgr.state.prompts[0].retry_count == 3
    assert mgr.state.failed_count == 1


def test_subprocess_kill_status_preserved_across_disk_roundtrip(mgr, mock_iface, tmp_path):  # FT-056
    """retry_count is persisted in YAML and survives a save → load roundtrip."""
    mock_iface.execute_prompt.return_value = _fail_result("killed")

    prompt = QueuedPrompt(
        id="t13aaaa", content="important work", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)
    mgr.state = None

    mgr._process_queue_iteration()

    reloaded = mgr.storage.load_queue_state()
    assert reloaded.prompts[0].status == PromptStatus.QUEUED
    assert reloaded.prompts[0].retry_count == 1


# ===========================================================================
# SIGKILL / Crash Recovery (T14–T23)
# ===========================================================================


class TestSIGKILLRecovery:
    """FT-057..066: filesystem state left after the entire queue daemon is killed.

    The defining artifact is an orphaned .executing.md file in the queue directory.
    """

    @pytest.fixture
    def mgr(self, tmp_path, mock_iface):
        """QueueManager with running=True (needed for shutdown tests)."""
        m = QueueManager.__new__(QueueManager)
        m.storage = QueueStorage(str(tmp_path))
        m.claude_interface = mock_iface
        m.check_interval = 30
        m.running = True
        m.state = None
        return m

    def test_orphaned_executing_file_present_after_sigkill_simulation(self, storage):  # FT-057
        """Baseline: writing a .executing.md creates the expected orphaned file."""
        exec_file = storage.queue_dir / "t14aaaa-do-work.executing.md"
        exec_file.write_text(
            "---\n"
            "priority: 0\n"
            "working_directory: .\n"
            "max_retries: 3\n"
            "status: executing\n"
            "retry_count: 1\n"
            "created_at: 2025-01-01T00:00:00\n"
            "---\n\n"
            "do work"
        )

        assert exec_file.exists() is True
        plain = storage.queue_dir / "t14aaaa-do-work.md"
        assert plain.exists() is False

    def test_sigkill_orphan_loaded_as_queued_not_executing(self, storage):  # FT-058
        """Orphaned .executing.md is loaded as QUEUED, not stuck EXECUTING."""
        _write_executing_file(storage.queue_dir, "t14aaaa", "do-work")

        state = storage.load_queue_state()

        assert len(state.prompts) == 1
        assert state.prompts[0].status == PromptStatus.QUEUED
        assert state.get_next_prompt() is not None

    def test_sigkill_orphan_recovered_as_queued_after_fix(self, storage):  # FT-059
        """A .executing.md left by a crash is loaded as status=QUEUED."""
        _write_executing_file(storage.queue_dir, "t14aaaa", "do-work")

        state = storage.load_queue_state()

        assert len(state.prompts) == 1
        assert state.prompts[0].status == PromptStatus.QUEUED
        assert state.prompts[0].id == "t14aaaa"
        assert state.get_next_prompt() is not None

    def test_sigkill_orphan_gets_recovery_log_entry(self, storage):  # FT-060
        """Recovered prompt has an execution_log entry containing 'ecovered'."""
        _write_executing_file(storage.queue_dir, "t14aaaa", "do-work")

        state = storage.load_queue_state()

        log = state.prompts[0].execution_log
        assert "ecovered" in log.lower() or "Recovered" in log, (
            f"Expected 'Recovered'/'ecovered' in log, got: {log!r}"
        )

    def test_multiple_sigkill_orphans_all_recovered(self, storage):  # FT-061
        """All .executing.md orphans are recovered as QUEUED on load."""
        pairs = [
            ("aaa11111", "task-one"),
            ("bbb22222", "task-two"),
            ("ccc33333", "task-three"),
        ]
        for pid, name in pairs:
            _write_executing_file(storage.queue_dir, pid, name, content=name)

        state = storage.load_queue_state()

        assert len(state.prompts) == 3
        assert all(p.status == PromptStatus.QUEUED for p in state.prompts)
        loaded_ids = {p.id for p in state.prompts}
        assert loaded_ids == {"aaa11111", "bbb22222", "ccc33333"}

    def test_sigkill_orphan_does_not_duplicate_with_plain_md(self, storage):  # FT-062
        """If both .executing.md and plain .md exist for the same ID, only one is loaded."""
        content = (
            "---\n"
            "priority: 0\n"
            "working_directory: .\n"
            "max_retries: 3\n"
            "status: queued\n"
            "retry_count: 0\n"
            "created_at: 2025-01-01T00:00:00\n"
            "---\n\n"
            "my task"
        )
        exec_file = storage.queue_dir / "t19aaaa-my-task.executing.md"
        plain_file = storage.queue_dir / "t19aaaa-my-task.md"
        exec_file.write_text(content)
        plain_file.write_text(content)

        state = storage.load_queue_state()

        copies = [p for p in state.prompts if p.id == "t19aaaa"]
        assert len(copies) == 1, (
            f"Expected exactly 1 copy of t19aaaa, found {len(copies)}"
        )

    def test_sigkill_with_corrupted_executing_file_skipped_gracefully(self, storage):  # FT-063
        """Corrupted YAML in a .executing.md does not raise; valid prompts still load."""
        bad_file = storage.queue_dir / "t20aaaa-bad-prompt.executing.md"
        bad_file.write_text("---\nthis: is: not: valid: yaml: [\n---\n\nbody")

        good_file = storage.queue_dir / "t21bbbb-good-prompt.md"
        good_file.write_text(
            "---\n"
            "priority: 0\n"
            "working_directory: .\n"
            "max_retries: 3\n"
            "status: queued\n"
            "retry_count: 0\n"
            "created_at: 2025-01-01T00:00:00\n"
            "---\n\n"
            "good task"
        )

        state = storage.load_queue_state()

        good_prompts = [p for p in state.prompts if p.id == "t21bbbb"]
        assert len(good_prompts) == 1, "Valid prompt was not loaded"

    def test_sigkill_orphan_is_dispatchable_after_fix(self, storage):  # FT-064
        """With crash recovery applied, orphaned .executing.md IS dispatchable."""
        exec_file = storage.queue_dir / "t21aaaa-stuck.executing.md"
        exec_file.write_text(
            "---\n"
            "priority: 0\n"
            "working_directory: .\n"
            "max_retries: 3\n"
            "status: executing\n"
            "retry_count: 0\n"
            "created_at: 2025-01-01T00:00:00\n"
            "---\n\n"
            "stuck task"
        )

        state = storage.load_queue_state()

        assert state.prompts[0].status == PromptStatus.QUEUED
        assert state.get_next_prompt() is not None

    def test_graceful_ctrl_c_shutdown_removes_executing_file(self, mgr):  # FT-065
        """_shutdown() transitions EXECUTING → QUEUED and removes the .executing.md file."""
        storage = mgr.storage

        prompt = QueuedPrompt(
            id="t22aaaa", content="active task", status=PromptStatus.EXECUTING
        )
        exec_path = storage.queue_dir / "t22aaaa-active-task.executing.md"
        MarkdownPromptParser.write_prompt_file(prompt, exec_path)
        assert exec_path.exists()

        state = QueueState()
        state.add_prompt(prompt)
        mgr.state = state

        mgr._shutdown()

        assert exec_path.exists() is False, ".executing.md must be removed by _shutdown()"
        plain_files = [
            f
            for f in storage.queue_dir.glob("t22aaaa*.md")
            if not f.name.endswith(".executing.md")
        ]
        assert len(plain_files) == 1, (
            f"Expected exactly one plain .md, found: {plain_files}"
        )
        assert mgr.state.prompts[0].status == PromptStatus.QUEUED

    def test_graceful_shutdown_multiple_in_flight_prompts(self, mgr):  # FT-066
        """All EXECUTING prompts become QUEUED and their .executing.md files are removed."""
        storage = mgr.storage

        prompts_cfg = [
            ("t23aaaa", "task-alpha"),
            ("t23bbbb", "task-beta"),
            ("t23cccc", "task-gamma"),
        ]

        exec_paths = []
        state = QueueState()
        for pid, slug in prompts_cfg:
            prompt = QueuedPrompt(
                id=pid, content=slug.replace("-", " "), status=PromptStatus.EXECUTING
            )
            exec_path = storage.queue_dir / f"{pid}-{slug}.executing.md"
            MarkdownPromptParser.write_prompt_file(prompt, exec_path)
            exec_paths.append(exec_path)
            state.add_prompt(prompt)

        mgr.state = state
        mgr._shutdown()

        for path in exec_paths:
            assert path.exists() is False, f"{path.name} must be removed by _shutdown()"

        for pid, _ in prompts_cfg:
            plain_files = [
                f
                for f in storage.queue_dir.glob(f"{pid}*.md")
                if not f.name.endswith(".executing.md")
            ]
            assert len(plain_files) == 1, (
                f"Expected one plain .md for {pid}, found: {plain_files}"
            )

        assert all(p.status == PromptStatus.QUEUED for p in mgr.state.prompts)


# ===========================================================================
# Network Failures (T24–T31)
# ===========================================================================


def test_network_timeout_during_execution_returns_failure(iface, mocker, tmp_path):  # FT-067
    """Hung network call → TimeoutExpired → caught as failure, not re-raised."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd=["claude", "--print", "..."], timeout=3600
    )

    prompt = QueuedPrompt(
        id="t24aaaa", content="fetch data", working_directory=str(tmp_path)
    )
    result = iface.execute_prompt(prompt)

    assert result.success is False
    assert result.is_rate_limited is False
    assert "timed out" in result.error.lower()


def test_network_failure_nonzero_returncode_not_rate_limited(iface, mocker, tmp_path):  # FT-068
    """returncode=1 with DNS error in stderr → failure, not rate-limited."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="curl: (6) Could not resolve host: api.anthropic.com",
    )

    prompt = QueuedPrompt(
        id="t25aaaa", content="some task", working_directory=str(tmp_path)
    )
    result = iface.execute_prompt(prompt)

    assert result.success is False
    assert result.is_rate_limited is False


@pytest.mark.parametrize(
    "output_string",
    [
        "curl: (6) Could not resolve host: api.anthropic.com",
        "connect: Connection timed out",
        "network unreachable",
        "ssl: handshake failure",
        "",
        "Killed",
    ],
)
def test_network_failure_output_does_not_match_rate_limit_keywords(iface, output_string):  # FT-069
    """Typical network-error strings are NOT classified as rate-limited."""
    info = iface._detect_rate_limit(output_string)
    assert info.is_rate_limited is False, (
        f"False positive: {output_string!r} incorrectly triggered rate-limit detection"
    )


def test_startup_connection_failure_prevents_queue_loop(tmp_path):  # FT-070
    """If test_connection() fails, start() exits before executing any prompts."""
    mock_iface = MagicMock(spec=ClaudeCodeInterface)
    mock_iface.test_connection.return_value = (
        False,
        "Claude Code CLI error: network unreachable",
    )

    manager = QueueManager.__new__(QueueManager)
    manager.storage = QueueStorage(str(tmp_path))
    manager.claude_interface = mock_iface
    manager.check_interval = 30
    manager.running = False
    manager.state = None

    manager.start()

    assert mock_iface.execute_prompt.call_count == 0
    assert manager.running is False


def test_verify_connection_timeout_returns_false(iface, mocker):  # FT-071
    """test_connection() TimeoutExpired → returns (False, '...timed out...')."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd=["claude", "--help"], timeout=10
    )

    ok, msg = iface.test_connection()

    assert ok is False
    assert "timed out" in msg.lower(), f"Expected 'timed out' in {msg!r}"


def test_network_loss_mid_execution_retries_after_fix(mgr, mock_iface):  # FT-072
    """Network failure with retries remaining → prompt back to QUEUED."""
    mock_iface.execute_prompt.return_value = _network_fail_result()

    prompt = QueuedPrompt(
        id="t29aaaa", content="fetch data", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)
    mgr.state = None

    mgr._process_queue_iteration()

    assert mgr.state.prompts[0].status == PromptStatus.QUEUED
    assert mgr.state.prompts[0].retry_count == 1


def test_repeated_network_failures_exhaust_retries(mgr, mock_iface):  # FT-073
    """Extended network outage: after max_retries the prompt is permanently FAILED."""
    mock_iface.execute_prompt.return_value = _network_fail_result()

    prompt = QueuedPrompt(
        id="t30aaaa", content="network task", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)

    for _ in range(3):
        mgr.state = None
        mgr._process_queue_iteration()

    assert mgr.state.prompts[0].status == PromptStatus.FAILED
    assert mgr.state.failed_count == 1


def test_network_restored_prompt_succeeds_on_retry(mgr, mock_iface):  # FT-074
    """First attempt fails (network error), second attempt succeeds → COMPLETED."""
    mock_iface.execute_prompt.side_effect = [
        _network_fail_result(),
        _success_result(),
    ]

    prompt = QueuedPrompt(
        id="t31aaaa", content="fetch data", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)

    mgr.state = None
    mgr._process_queue_iteration()
    assert mgr.state.prompts[0].status == PromptStatus.QUEUED

    mgr.state = None
    mgr._process_queue_iteration()

    assert mgr.state.total_processed == 1


# ===========================================================================
# Authentication Failures (T32–T40)
# ===========================================================================


@pytest.mark.parametrize(
    "error_string",
    [
        "Error: Not authenticated. Please run 'claude login'.",
        "authentication failed",
        "Unauthorized: invalid API key",
        "Error: API key is invalid or has been revoked",
        "403 Forbidden",
        "401 Unauthorized",
        "Error: Your session has expired. Please log in again.",
    ],
)
def test_auth_error_text_not_classified_as_rate_limit(iface, error_string):  # FT-075
    """Authentication error messages do NOT match rate-limit indicators."""
    info = iface._detect_rate_limit(error_string)
    assert info.is_rate_limited is False, (
        f"False positive: auth string {error_string!r} triggered rate-limit detection"
    )


def test_auth_failure_returncode_nonzero_produces_failure_result(iface, mocker, tmp_path):  # FT-076
    """returncode=1 with auth error in stderr → success=False, is_rate_limited=False."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="",
        stderr="Error: Not authenticated. Please run 'claude login'.",
    )

    prompt = QueuedPrompt(
        id="t33aaaa", content="task", working_directory=str(tmp_path)
    )
    result = iface.execute_prompt(prompt)

    assert result.success is False
    assert result.is_rate_limited is False
    assert "Not authenticated" in result.error


def test_verify_claude_unavailable_raises_runtime_error(mocker):  # FT-077
    """FileNotFoundError from subprocess during _verify_claude_available → RuntimeError."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.side_effect = FileNotFoundError("No such file or directory: 'claude'")

    with pytest.raises(RuntimeError) as exc_info:
        ClaudeCodeInterface("claude", 60)

    msg = str(exc_info.value).lower()
    assert "not found" in msg or "path" in msg, (
        f"Expected 'not found'/'path' in error: {exc_info.value}"
    )


def test_verify_claude_version_nonzero_raises_runtime_error(mocker):  # FT-078
    """Non-zero returncode from claude --version during __init__ → RuntimeError."""
    mock_run = mocker.patch("claude_code_queue.claude_interface.subprocess.run")
    mock_run.return_value = MagicMock(
        returncode=1, stdout="", stderr="configuration error"
    )

    with pytest.raises(RuntimeError):
        ClaudeCodeInterface("claude", 60)


def test_startup_auth_failure_prevents_start(tmp_path):  # FT-079
    """If test_connection() returns auth failure, start() exits without executing."""
    mock_iface = MagicMock(spec=ClaudeCodeInterface)
    mock_iface.test_connection.return_value = (
        False,
        "Claude Code CLI error: Error: Not authenticated.",
    )

    manager = QueueManager.__new__(QueueManager)
    manager.storage = QueueStorage(str(tmp_path))
    manager.claude_interface = mock_iface
    manager.check_interval = 30
    manager.running = False
    manager.state = None

    manager.start()

    assert mock_iface.execute_prompt.call_count == 0
    assert manager.running is False


def test_wrong_credentials_first_attempt_fails(mgr, mock_iface):  # FT-080
    """Wrong API key: first failure with retries remaining → QUEUED."""
    mock_iface.execute_prompt.return_value = _auth_fail_result(
        "Unauthorized: invalid API key"
    )

    prompt = QueuedPrompt(
        id="t37aaaa", content="task", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)
    mgr.state = None

    mgr._process_queue_iteration()

    assert mgr.state.prompts[0].status == PromptStatus.QUEUED
    assert mgr.state.prompts[0].retry_count == 1
    assert mgr.state.failed_count == 0


def test_wrong_credentials_exhaust_retries_marks_failed(mgr, mock_iface):  # FT-081
    """Wrong credentials: after max_retries the prompt is permanently FAILED."""
    mock_iface.execute_prompt.return_value = _auth_fail_result(
        "Unauthorized: invalid API key"
    )

    prompt = QueuedPrompt(
        id="t38aaaa", content="task", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)

    for _ in range(3):
        mgr.state = None
        mgr._process_queue_iteration()

    assert mgr.state.prompts[0].status == PromptStatus.FAILED
    assert mgr.state.failed_count == 1
    assert mgr.state.prompts[0].retry_count == 3


def test_interrupted_auth_session_mid_queue(mgr, mock_iface):  # FT-082
    """Token expires between tasks: first COMPLETED, second QUEUED for retry."""
    mock_iface.execute_prompt.side_effect = [
        _success_result("done"),
        _auth_fail_result("session expired"),
    ]

    prompt1 = QueuedPrompt(
        id="t39aaaa", content="task one", priority=0, max_retries=3
    )
    prompt2 = QueuedPrompt(
        id="t39bbbb", content="task two", priority=1, max_retries=3
    )
    state = QueueState()
    state.add_prompt(prompt1)
    state.add_prompt(prompt2)
    mgr.storage.save_queue_state(state)

    mgr.state = None
    mgr._process_queue_iteration()
    assert mgr.state.total_processed == 1

    mgr.state = None
    mgr._process_queue_iteration()

    p2 = mgr.state.prompts[0]
    assert p2.id == "t39bbbb"
    assert p2.status == PromptStatus.QUEUED
    assert mgr.state.total_processed == 1
    assert mgr.state.failed_count == 0


def test_auth_failure_not_rate_limit_counter_not_incremented(mgr, mock_iface):  # FT-083
    """Auth failure must not increment state.rate_limited_count."""
    mock_iface.execute_prompt.return_value = _auth_fail_result()

    prompt = QueuedPrompt(
        id="t40aaaa", content="task", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    mgr.storage.save_queue_state(state)
    mgr.state = None

    mgr._process_queue_iteration()

    assert mgr.state.rate_limited_count == 0


# ===========================================================================
# Complex Failure Scenarios (T41–T42)
# ===========================================================================


def test_sigkill_during_auth_failure_retry_cycle(tmp_path):  # FT-084
    """Prompt fails with auth error → QUEUED for retry. Before the next tick,
    process is killed (simulated by writing .executing.md with retry_count=1).
    On restart, the orphan is recovered as QUEUED with retry_count=1 intact.
    """
    storage = QueueStorage(str(tmp_path))

    exec_file = storage.queue_dir / "t41aaaa-auth-task.executing.md"
    exec_file.write_text(
        "---\n"
        "priority: 0\n"
        "working_directory: .\n"
        "max_retries: 3\n"
        "status: executing\n"
        "retry_count: 1\n"
        "created_at: 2025-01-01T00:00:00\n"
        "---\n\n"
        "auth task"
    )

    state = storage.load_queue_state()

    assert len(state.prompts) == 1
    p = state.prompts[0]
    assert p.status == PromptStatus.QUEUED
    assert p.retry_count == 1
    assert p.can_retry() is True


def test_network_loss_then_sigkill_then_restart_full_recovery(tmp_path, mock_iface):  # FT-085
    """Full scenario:
    1. Prompt queued.
    2. First execution: network failure.
    3. Prompt goes to QUEUED for retry (retry_count=1).
    4. SIGKILL before second execution completes
       (.executing.md with retry_count=2 left on disk).
    5. Restart: prompt recovered as QUEUED with retry_count=2.
    6. Second execution succeeds → COMPLETED.
    """
    storage = QueueStorage(str(tmp_path))

    mock_iface.execute_prompt.return_value = _fail_result(
        "curl: (6) Could not resolve host: api.anthropic.com"
    )

    mgr1 = QueueManager.__new__(QueueManager)
    mgr1.storage = storage
    mgr1.claude_interface = mock_iface
    mgr1.check_interval = 30
    mgr1.running = False
    mgr1.state = None

    prompt = QueuedPrompt(
        id="t42aaaa", content="fetch data", max_retries=3, retry_count=0
    )
    state = QueueState()
    state.add_prompt(prompt)
    storage.save_queue_state(state)

    mgr1._process_queue_iteration()

    p = mgr1.state.prompts[0]
    assert p.status == PromptStatus.QUEUED
    assert p.retry_count == 1

    for f in storage.queue_dir.glob("t42aaaa*.md"):
        f.unlink()

    exec_file = storage.queue_dir / "t42aaaa-fetch-data.executing.md"
    exec_file.write_text(
        "---\n"
        "priority: 0\n"
        "working_directory: .\n"
        "max_retries: 3\n"
        "status: executing\n"
        "retry_count: 2\n"
        "created_at: 2025-01-01T00:00:00\n"
        "---\n\n"
        "fetch data"
    )

    reloaded_state = storage.load_queue_state()

    assert len(reloaded_state.prompts) == 1
    recovered = reloaded_state.prompts[0]
    assert recovered.status == PromptStatus.QUEUED
    assert recovered.retry_count == 2
    assert recovered.can_retry() is True

    mock_iface.execute_prompt.return_value = _success_result()

    mgr2 = QueueManager.__new__(QueueManager)
    mgr2.storage = storage
    mgr2.claude_interface = mock_iface
    mgr2.check_interval = 30
    mgr2.running = False
    mgr2.state = None

    mgr2._process_queue_iteration()

    assert mgr2.state.total_processed == 1

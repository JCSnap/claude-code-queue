"""
Shared fixtures for claude-code-queue test suite.
"""

import pytest
from claude_code_queue.models import QueuedPrompt, QueueState, PromptStatus
from claude_code_queue.storage import QueueStorage
from claude_code_queue.claude_interface import ClaudeCodeInterface
from claude_code_queue.queue_manager import QueueManager


@pytest.fixture
def storage(tmp_path):
    """QueueStorage backed by a fresh temporary directory."""
    return QueueStorage(str(tmp_path))


@pytest.fixture
def interface(mocker):
    """ClaudeCodeInterface with _verify_claude_available patched out."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    return ClaudeCodeInterface(claude_command="claude", timeout=60)


@pytest.fixture
def manager(tmp_path, mocker):
    """QueueManager backed by a fresh temporary directory, with Claude patched."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    mocker.patch.object(
        ClaudeCodeInterface, "test_connection", return_value=(True, "ok")
    )
    return QueueManager(storage_dir=str(tmp_path), claude_command="claude")


@pytest.fixture
def sample_prompt():
    """A QueuedPrompt with a known id and content for use in tests."""
    return QueuedPrompt(id="abc12345", content="test prompt content")

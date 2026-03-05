"""
Shared fixtures for claude-code-queue test suite.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "llm_eval: marks tests that call the Anthropic API (requires ANTHROPIC_API_KEY)",
    )


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
    iface = ClaudeCodeInterface(claude_command="claude", timeout=60)
    yield iface
    iface.close()


@pytest.fixture
def manager(tmp_path, mocker):
    """QueueManager backed by a fresh temporary directory, with Claude patched."""
    mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
    mocker.patch.object(
        ClaudeCodeInterface, "test_connection", return_value=(True, "ok")
    )
    mgr = QueueManager(storage_dir=str(tmp_path), claude_command="claude")
    yield mgr
    mgr.claude_interface.close()


@pytest.fixture
def sample_prompt():
    """A QueuedPrompt with a known id and content for use in tests."""
    return QueuedPrompt(id="abc12345", content="test prompt content")

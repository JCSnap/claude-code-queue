"""
Integration and LLM evaluation tests for the bundled Claude Code skill — Pass SK-2.

Covers:
  - Queue round-trip: write a file in SKILL.md template format, process via
    QueueManager with mocked Claude, verify completion. (no API required)
  - LLM eval: real Anthropic API calls verifying Claude behaves correctly
    when the skill content is loaded as a system prompt. (requires
    ANTHROPIC_API_KEY; tests are skipped when the key is absent)

Run just the free tests:
  pytest tests/test_skill_integration.py -m "not llm_eval" -v

Run all tests (requires API key):
  ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_skill_integration.py -v
"""

import os
import re
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from claude_code_queue.claude_interface import ClaudeCodeInterface
from claude_code_queue.models import ExecutionResult, QueuedPrompt
from claude_code_queue.queue_manager import QueueManager
from claude_code_queue.storage import MarkdownPromptParser, QueueStorage


# ---------------------------------------------------------------------------
# Shared constants and fixtures
# ---------------------------------------------------------------------------

SKILL_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "claude_code_queue"
    / "skills"
    / "queue"
    / "SKILL.md"
)

# Cheapest model for eval — deterministic enough for structural assertions.
_EVAL_MODEL = "claude-haiku-4-5-20251001"

# Decorator: skip test when ANTHROPIC_API_KEY is not present in the environment.
llm_eval = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping LLM eval test",
)


@pytest.fixture(scope="module")
def skill_content():
    """Full text of the bundled SKILL.md (system prompt for eval tests)."""
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def anthropic_client():
    """Anthropic SDK client (module-scoped; stateless, safe to share)."""
    import anthropic  # optional dependency — only imported when needed
    return anthropic.Anthropic()


# ---------------------------------------------------------------------------
# SI-001: Queue round-trip (no API required)
# ---------------------------------------------------------------------------

class TestQueueRoundTrip:
    """
    Verify that a prompt file written in the exact format shown in SKILL.md
    is correctly ingested, executed (with mocked Claude), and completed.
    """

    def test_si001_skill_template_format_submits_and_completes(
        self, tmp_path, mocker
    ):
        """
        SI-001: SKILL.md template format → queue → execute → completed dir.

        This is the core integration test: it writes a raw .md file using the
        same frontmatter schema documented in SKILL.md, then verifies the
        QueueManager picks it up via _load_prompts_from_files and moves it to
        the completed directory after a successful (mocked) execution.
        """
        mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")
        mocker.patch.object(
            ClaudeCodeInterface, "test_connection", return_value=(True, "ok")
        )

        # Write a queue file using the exact frontmatter format from SKILL.md.
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir(parents=True)
        queue_file = queue_dir / "testid-fix-auth.md"
        queue_file.write_text(
            f"---\n"
            f"priority: 1\n"
            f"working_directory: {tmp_path}\n"
            f"context_files: []\n"
            f"max_retries: 3\n"
            f"estimated_tokens: null\n"
            f"---\n\n"
            f"# Fix authentication bug\n\n"
            f"Locate and fix the login bug in the auth module.\n",
            encoding="utf-8",
        )

        manager = QueueManager(
            storage_dir=str(tmp_path), claude_command="claude"
        )
        mock_result = ExecutionResult(
            success=True,
            output="Fixed the auth bug successfully.",
            error="",
            execution_time=0.5,
        )

        with patch.object(
            manager.claude_interface, "execute_prompt", return_value=mock_result
        ):
            manager._process_queue_iteration()

        completed = list((tmp_path / "completed").glob("*.md"))
        assert len(completed) == 1, (
            "Expected one completed prompt; "
            f"found {len(completed)} in {tmp_path / 'completed'}"
        )

    def test_si002_installed_skill_content_is_readable_yaml(self, tmp_path):
        """
        SI-002: After install-skill writes the file, it can be re-parsed as
        valid YAML frontmatter — a sanity check on the file-copy path.
        """
        from pathlib import Path as _Path
        import claude_code_queue.cli as cli_mod

        dest = tmp_path / ".claude" / "skills" / "queue" / "SKILL.md"

        with patch.object(_Path, "home", return_value=tmp_path):
            with patch("sys.argv", ["claude-queue", "install-skill"]):
                from claude_code_queue.cli import main
                main()

        assert dest.exists()
        text = dest.read_text(encoding="utf-8")
        parts = text.split("---\n", 2)
        assert len(parts) >= 3
        parsed = yaml.safe_load(parts[1])
        assert parsed["name"] == "queue"
        assert "Bash" in " ".join(
            parsed["allowed-tools"]
            if isinstance(parsed["allowed-tools"], list)
            else [parsed["allowed-tools"]]
        )


# ---------------------------------------------------------------------------
# LLM eval tests (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

class TestSkillLLMEval:
    """
    Evaluate Claude's behaviour when the SKILL.md content is provided as a
    system prompt.  These tests make real Anthropic API calls and are skipped
    when ANTHROPIC_API_KEY is not present in the environment.

    Assertions are structural rather than exact-match to accommodate the
    non-deterministic nature of LLM outputs.
    """

    @llm_eval
    def test_si003_claude_suggests_queuing_for_complex_task(
        self, anthropic_client, skill_content
    ):
        """
        SI-003: Claude mentions claude-queue when prompted with a task likely
        to hit API rate limits.
        """
        response = anthropic_client.messages.create(
            model=_EVAL_MODEL,
            max_tokens=512,
            system=skill_content,
            messages=[{
                "role": "user",
                "content": (
                    "I need to refactor the authentication code across 15 files "
                    "to use the new OAuth2 library. This will be a very large change."
                ),
            }],
        )
        text = response.content[0].text.lower()
        assert "claude-queue" in text or "queue" in text, (
            f"Expected Claude to mention queuing for a complex multi-file task. "
            f"Response: {response.content[0].text!r}"
        )

    @llm_eval
    def test_si004_claude_produces_valid_yaml_frontmatter(
        self, anthropic_client, skill_content
    ):
        """
        SI-004: Claude generates a queue template with valid, parseable YAML
        frontmatter when asked to create a prompt file.
        """
        response = anthropic_client.messages.create(
            model=_EVAL_MODEL,
            max_tokens=1024,
            system=skill_content,
            messages=[{
                "role": "user",
                "content": (
                    "Please create a queue template to fix the login bug in "
                    "/home/user/myapp"
                ),
            }],
        )
        text = response.content[0].text

        match = re.search(r"^---\n(.*?)\n---", text, re.DOTALL | re.MULTILINE)
        assert match, (
            "Expected YAML frontmatter between --- markers in Claude's response. "
            f"Response: {text!r}"
        )

        parsed = yaml.safe_load(match.group(1))
        assert "priority" in parsed, "Template should include 'priority' field"
        assert "working_directory" in parsed, "Template should include 'working_directory'"
        assert "max_retries" in parsed, "Template should include 'max_retries'"

    @llm_eval
    def test_si005_claude_template_parses_with_storage(
        self, anthropic_client, skill_content, tmp_path
    ):
        """
        SI-005: The template Claude produces is accepted by MarkdownPromptParser
        without error — the ultimate consistency check.
        """
        response = anthropic_client.messages.create(
            model=_EVAL_MODEL,
            max_tokens=1024,
            system=skill_content,
            messages=[{
                "role": "user",
                "content": (
                    f"Create a queue template to fix the auth bug. "
                    f"Use working_directory: {tmp_path}"
                ),
            }],
        )
        text = response.content[0].text

        # Accept the template whether Claude wraps it in ```markdown or not.
        md_match = re.search(
            r"```(?:markdown)?\n(---\n.*?---\n.*?)```", text, re.DOTALL
        )
        content = md_match.group(1) if md_match else text

        template_file = tmp_path / "testid-claude-generated.md"
        template_file.write_text(content, encoding="utf-8")

        parser = MarkdownPromptParser()
        prompt = parser.parse_prompt_file(template_file)

        assert prompt is not None, (
            f"MarkdownPromptParser could not parse Claude's generated template. "
            f"Template content:\n{content}"
        )
        assert prompt.priority is not None

    @llm_eval
    def test_si006_claude_generated_add_command_submits_to_queue(
        self, anthropic_client, skill_content, tmp_path
    ):
        """
        SI-006: Claude generates a 'claude-queue add' command; when executed
        against a temp storage directory, the job appears in the queue.

        This is the end-to-end test: LLM output → shell command → queue file.
        """
        response = anthropic_client.messages.create(
            model=_EVAL_MODEL,
            max_tokens=256,
            system=skill_content,
            messages=[{
                "role": "user",
                "content": (
                    f"Add a task to the queue to fix the login bug. "
                    f"Use --working-dir {tmp_path}. "
                    f"Output ONLY the claude-queue add command, nothing else."
                ),
            }],
        )
        text = response.content[0].text.strip()

        # Accept commands inside a ```bash fence or as plain text.
        bash_match = re.search(r"```(?:bash)?\n(claude-queue add.*?)```", text, re.DOTALL)
        plain_match = re.search(r"claude-queue add\s+[^\n]+", text)
        cmd_str = (
            bash_match.group(1).strip()
            if bash_match
            else (plain_match.group(0) if plain_match else None)
        )
        assert cmd_str, (
            f"Expected a 'claude-queue add ...' command in the response. "
            f"Got: {text!r}"
        )

        # Replace 'claude-queue' with the Python module invocation so the command
        # runs against the local source tree without needing an installed entry-point.
        parts = shlex.split(cmd_str)
        parts[0:1] = [sys.executable, "-m", "claude_code_queue.cli"]
        parts += ["--storage-dir", str(tmp_path)]

        import subprocess
        result = subprocess.run(parts, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"claude-queue add command failed.\n"
            f"Command: {parts}\n"
            f"stderr: {result.stderr}"
        )

        storage = QueueStorage(str(tmp_path))
        state = storage.load_queue_state()
        assert len(state.prompts) >= 1, (
            "Expected at least one queued prompt after running the generated command."
        )
        assert state.prompts[0].status.value == "queued"

"""
Tests for the bundled Claude Code skill — Pass SK-1.

Covers:
  - SKILL.md structural validity (YAML frontmatter, required fields)
  - Consistency between skill documentation and the actual CLI / data model
  - install-skill CLI command (file creation, idempotency, --force, error paths)
"""

import dataclasses
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import claude_code_queue.cli as _cli_module
from claude_code_queue.cli import main
from claude_code_queue.models import QueuedPrompt
from claude_code_queue.storage import MarkdownPromptParser


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SKILL_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "claude_code_queue"
    / "skills"
    / "queue"
    / "SKILL.md"
)

BATCH_SKILL_PATH = SKILL_PATH.parent.parent / "batch-wizard" / "SKILL.md"

# Known CLI subcommands registered in cli.py.
_CLI_SUBCOMMANDS = {
    "start", "add", "template", "status", "cancel",
    "list", "test", "bank", "batch", "install-skill", "prompt-box",
    "resume-session", "sessions",
}


@pytest.fixture(scope="module")
def skill_text():
    """Full text of the bundled SKILL.md."""
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def skill_frontmatter(skill_text):
    """Parsed YAML from the SKILL.md frontmatter block."""
    parts = skill_text.split("---\n", 2)
    assert len(parts) >= 3, "SKILL.md must open with a YAML frontmatter block"
    return yaml.safe_load(parts[1])


# ---------------------------------------------------------------------------
# SK-001 – SK-005: SKILL.md structural validity
# ---------------------------------------------------------------------------

class TestSkillFileStructure:
    """Verify the bundled SKILL.md is well-formed and contains required fields."""

    def test_sk001_skill_md_exists_in_package(self):
        """SK-001: SKILL.md is present at the expected package path."""
        assert SKILL_PATH.exists(), (
            f"Bundled SKILL.md not found at {SKILL_PATH}. "
            "Was 'skills/**/*' added to pyproject.toml package-data?"
        )

    def test_sk002_frontmatter_is_valid_yaml(self, skill_text):
        """SK-002: The YAML frontmatter block parses without error."""
        parts = skill_text.split("---\n", 2)
        assert len(parts) >= 3, "SKILL.md must have an opening --- block"
        yaml.safe_load(parts[1])  # raises yaml.YAMLError on invalid YAML

    def test_sk003_required_field_name(self, skill_frontmatter):
        """SK-003: 'name' field is present and set to 'queue'."""
        assert "name" in skill_frontmatter
        assert skill_frontmatter["name"] == "queue"

    def test_sk004_required_field_description(self, skill_frontmatter):
        """SK-004: 'description' field is present and non-trivial."""
        assert "description" in skill_frontmatter
        assert len(str(skill_frontmatter["description"])) > 20

    def test_sk005_allowed_tools_includes_bash(self, skill_frontmatter):
        """SK-005: 'allowed-tools' includes Bash so Claude can run claude-queue."""
        assert "allowed-tools" in skill_frontmatter
        tools = skill_frontmatter["allowed-tools"]
        tool_str = " ".join(tools) if isinstance(tools, list) else str(tools)
        assert "Bash" in tool_str


    def test_batch_wizard_has_minimal_required_tools(self):
        """The wizard declares only the tools used by its workflow."""
        parts = BATCH_SKILL_PATH.read_text(encoding="utf-8").split("---\n", 2)
        frontmatter = yaml.safe_load(parts[1])

        assert frontmatter["name"] == "batch-wizard"
        assert frontmatter["allowed-tools"] == [
            "Bash", "Read", "Glob", "Grep", "Write", "Edit", "Agent"
        ]


# ---------------------------------------------------------------------------
# SK-006 – SK-007: Consistency between SKILL.md and the codebase
# ---------------------------------------------------------------------------

class TestSkillConsistency:
    """Verify SKILL.md documentation matches the real CLI and data model."""

    def test_sk006_commands_in_skill_match_cli_subparsers(self, skill_text):
        """SK-006: Every 'claude-queue <sub>' in bash blocks is a real CLI subcommand."""
        bash_blocks = re.findall(r"```bash\n(.*?)```", skill_text, re.DOTALL)
        mentioned = set()
        for block in bash_blocks:
            for line in block.splitlines():
                line = line.strip().lstrip("# ")
                m = re.match(r"claude-queue\s+([\w-]+)", line)
                if m:
                    mentioned.add(m.group(1))

        for sub in mentioned:
            assert sub in _CLI_SUBCOMMANDS, (
                f"Command 'claude-queue {sub}' appears in SKILL.md "
                f"but is not a registered CLI subcommand. "
                f"Known subcommands: {sorted(_CLI_SUBCOMMANDS)}"
            )

    def test_sk007_template_frontmatter_fields_match_queued_prompt(self, skill_text):
        """SK-007: YAML keys in the template block map to QueuedPrompt field names."""
        # Find the first ```markdown block that starts with frontmatter
        md_blocks = re.findall(r"```markdown\n(---\n.*?---)", skill_text, re.DOTALL)
        assert md_blocks, "No ```markdown block with frontmatter found in SKILL.md"

        # Captured block is "---\nkeys\n---"; extract just the keys section.
        template_yaml_str = md_blocks[0].split("---\n", 2)[1].split("\n---")[0]
        template_yaml = yaml.safe_load(template_yaml_str)
        assert template_yaml, "Template frontmatter parsed as empty"

        prompt_fields = {f.name for f in dataclasses.fields(QueuedPrompt)}
        for key in template_yaml:
            assert key in prompt_fields, (
                f"Template field '{key}' in SKILL.md is not a QueuedPrompt field. "
                f"Valid fields: {sorted(prompt_fields)}"
            )


# ---------------------------------------------------------------------------
# SK-008: Template round-trip through storage parser
# ---------------------------------------------------------------------------

class TestSkillTemplateParseable:
    """Verify the SKILL.md prompt template is accepted by the real storage parser."""

    def test_sk008_template_parses_without_error(self, skill_text, tmp_path):
        """SK-008: A file written in the SKILL.md template format parses correctly."""
        # Extract the first full ```markdown block (frontmatter + body)
        match = re.search(
            r"```markdown\n(---\n.*?---\n.*?)```",
            skill_text,
            re.DOTALL,
        )
        assert match, "No complete ```markdown template block found in SKILL.md"

        # Substitute the placeholder working_directory with a real path
        content = match.group(1).replace(
            "/absolute/path/to/project", str(tmp_path)
        )
        # Remove placeholder context_files entries (parser accepts but may warn)
        content = re.sub(
            r"context_files:\n(  - .*\n)+",
            "context_files: []\n",
            content,
        )

        template_file = tmp_path / "testid-skill-template.md"
        template_file.write_text(content, encoding="utf-8")

        parser = MarkdownPromptParser()
        prompt = parser.parse_prompt_file(template_file)

        assert prompt is not None, "MarkdownPromptParser returned None for a valid template"
        assert prompt.priority == 0
        assert prompt.max_retries == 3
        assert prompt.working_directory == str(tmp_path)


# ---------------------------------------------------------------------------
# SK-009 – SK-018: install-skill CLI command
# ---------------------------------------------------------------------------

@pytest.fixture
def skill_home(tmp_path, monkeypatch):
    """
    Redirect Path.home() → tmp_path for the duration of one test.

    install-skill writes to <claude config dir> / "skills" / "queue" / "SKILL.md".
    By redirecting home — and clearing CLAUDE_CONFIG_DIR so the developer's own
    profile selection cannot leak in — all file operations land in the test's
    temporary directory without touching a real profile.
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


class TestInstallSkillCommand:
    """Unit tests for 'claude-queue install-skill'."""

    @staticmethod
    def _run(*extra_args):
        """Invoke main() with sys.argv configured for install-skill."""
        with patch("sys.argv", ["claude-queue", "install-skill"] + list(extra_args)):
            return main()

    @staticmethod
    def _dest(home: Path, name: str = "queue") -> Path:
        return home / ".claude" / "skills" / name / "SKILL.md"

    def test_sk019_installs_into_the_active_profile(self, tmp_path, monkeypatch):
        """SK-019: the destination follows CLAUDE_CONFIG_DIR.

        Running several Claude Code profiles means several skills/ directories.
        Hardcoding ~/.claude installs into whichever profile is not in use.
        """
        profile = tmp_path / ".claude-work-2"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(profile))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert self._run() == 0
        assert (profile / "skills" / "queue" / "SKILL.md").exists()
        assert not self._dest(tmp_path).exists()

    def test_sk020_profiles_install_independently(self, tmp_path, monkeypatch):
        """SK-020: installing into one profile leaves the others alone."""
        first, second = tmp_path / "p1", tmp_path / "p2"
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(first))
        assert self._run() == 0
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(second))
        assert self._run() == 0

        assert (first / "skills" / "queue" / "SKILL.md").exists()
        assert (second / "skills" / "queue" / "SKILL.md").exists()

    def test_sk009_fresh_install_returns_0(self, skill_home):
        """SK-009: First installation exits with code 0."""
        assert self._run() == 0

    def test_sk010_fresh_install_creates_file(self, skill_home):
        """SK-010: SKILL.md is written to ~/.claude/skills/queue/."""
        self._run()
        assert self._dest(skill_home).exists()

    def test_sk011_fresh_install_creates_parent_dirs(self, skill_home):
        """SK-011: Intermediate directories are created as needed."""
        self._run()
        assert (skill_home / ".claude" / "skills" / "queue").is_dir()

    def test_sk012_fresh_install_content_matches_bundled(self, skill_home):
        """SK-012: Written file content matches the bundled SKILL.md exactly."""
        self._run()
        written = self._dest(skill_home).read_text(encoding="utf-8")
        bundled = SKILL_PATH.read_text(encoding="utf-8")
        assert written == bundled

    def test_sk013_fresh_install_prints_destination(self, skill_home, capsys):
        """SK-013: Success message includes the destination path."""
        self._run()
        out = capsys.readouterr().out
        assert str(self._dest(skill_home)) in out

    def test_sk014_already_installed_returns_1(self, skill_home):
        """SK-014: A second install (without --force) exits with code 1."""
        self._run()
        assert self._run() == 1

    def test_sk015_already_installed_warns_user(self, skill_home, capsys):
        """SK-015: Warning message mentions 'already installed' and '--force'."""
        self._run()
        capsys.readouterr()  # clear first install output
        self._run()
        out = capsys.readouterr().out
        assert "already installed" in out.lower()
        assert "--force" in out

    def test_sk016_force_overwrites_returns_0(self, skill_home):
        """SK-016: --force on an existing install exits with code 0."""
        self._run()
        assert self._run("--force") == 0

    def test_sk017_force_rewrites_with_bundled_content(self, skill_home):
        """SK-017: --force replaces a modified file with the bundled version."""
        dest = self._dest(skill_home)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("stale content", encoding="utf-8")

        self._run("--force")

        assert dest.read_text(encoding="utf-8") == SKILL_PATH.read_text(encoding="utf-8")

    def test_sk018_missing_bundled_source_returns_1(self, skill_home, monkeypatch):
        """SK-018: Returns 1 and prints an error if the bundled SKILL.md is missing."""
        # Point the module's __file__ to a dir that has no skills/ alongside it
        monkeypatch.setattr(_cli_module, "__file__", str(skill_home / "cli.py"))
        assert self._run() == 1

    def test_sk018b_missing_bundled_source_prints_error(
        self, skill_home, monkeypatch, capsys
    ):
        """SK-018b: Error message is printed when bundled source is missing."""
        monkeypatch.setattr(_cli_module, "__file__", str(skill_home / "cli.py"))
        self._run()
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "error" in out.lower()
    def test_default_install_writes_every_bundled_skill(self, skill_home):
        assert self._run() == 0
        assert self._dest(skill_home, "queue").exists()
        assert self._dest(skill_home, "batch-wizard").exists()

    @pytest.mark.parametrize("name", ["queue", "batch-wizard"])
    def test_named_install_writes_only_selected_skill(self, skill_home, name):
        assert self._run(name) == 0
        assert self._dest(skill_home, name).exists()
        other = "batch-wizard" if name == "queue" else "queue"
        assert not self._dest(skill_home, other).exists()

    def test_unknown_skill_returns_stable_error(self, skill_home, capsys):
        assert self._run("missing") == 1
        output = capsys.readouterr().out
        assert "unknown bundled skill 'missing'" in output
        assert "batch-wizard, queue" in output

    def test_existing_skill_prevents_partial_install(self, skill_home):
        queue_dest = self._dest(skill_home)
        queue_dest.parent.mkdir(parents=True)
        queue_dest.write_text("custom", encoding="utf-8")

        assert self._run() == 1
        assert queue_dest.read_text(encoding="utf-8") == "custom"
        assert not self._dest(skill_home, "batch-wizard").exists()

    def test_force_updates_every_bundled_skill(self, skill_home):
        for name in ("queue", "batch-wizard"):
            dest = self._dest(skill_home, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("stale", encoding="utf-8")

        assert self._run("--force") == 0
        assert self._dest(skill_home).read_text(
            encoding="utf-8"
        ) == SKILL_PATH.read_text(encoding="utf-8")
        assert self._dest(skill_home, "batch-wizard").read_text(
            encoding="utf-8"
        ) == BATCH_SKILL_PATH.read_text(encoding="utf-8")

    def test_write_failure_rolls_back_new_files(self, skill_home):
        original_write_text = Path.write_text

        def fail_batch_write(path, data, *args, **kwargs):
            if path.parent.name == "batch-wizard":
                raise OSError("disk full")
            return original_write_text(path, data, *args, **kwargs)

        with patch.object(Path, "write_text", autospec=True, side_effect=fail_batch_write):
            assert self._run() == 1

        assert not self._dest(skill_home).exists()
        assert not self._dest(skill_home, "batch-wizard").exists()

    def test_relative_active_profile_is_resolved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "profile")

        assert self._run("batch-wizard") == 0
        assert (tmp_path / "profile" / "skills" / "batch-wizard" / "SKILL.md").exists()

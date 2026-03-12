"""
Queue storage system with markdown support.
"""

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import yaml  # type: ignore

from .models import QueuedPrompt, QueueState, PromptStatus


class MarkdownPromptParser:
    """Parser for markdown-based prompt files."""

    @staticmethod
    def parse_prompt_file(file_path: Path) -> Optional[QueuedPrompt]:
        """Parse a markdown prompt file into a QueuedPrompt object."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if content.startswith("---\n"):
                parts = content.split("---\n", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    markdown_content = parts[2].strip()
                else:
                    frontmatter = ""
                    markdown_content = content
            else:
                frontmatter = ""
                markdown_content = content

            metadata: dict = {}
            if frontmatter.strip():
                try:
                    metadata = yaml.safe_load(frontmatter) or {}
                except yaml.YAMLError as e:
                    print(
                        f"Warning: YAML parse error in {file_path}, using defaults: {e}",
                        file=sys.stderr,
                    )
                    metadata = {}

            # R6 / Fix 3b — Strip execution log from content before assigning to prompt.
            #
            # Primary: split on the unique HTML sentinel (robust, no collision risk).
            # Fallback: legacy "## Execution Log" format written before sentinel was
            # introduced.
            #
            # WARNING — separator collision: if the user's prompt content contains the
            # literal separator string, the split will fire early. This is an inherent
            # limitation of the inline-separator markdown format. maxsplit=1 prevents
            # multiple splits but cannot distinguish content-embedded separators.
            SENTINEL = "\n\n<!-- claude-queue:execution-log -->"
            LOG_SEPARATOR = "\n\n## Execution Log\n"

            if SENTINEL in markdown_content:
                # Primary path: split on the HTML sentinel.
                prompt_content = markdown_content.split(SENTINEL, 1)[0].strip()
                prompt_execution_log = ""  # log section not loaded back into memory
            elif LOG_SEPARATOR in markdown_content:
                # Fallback path: legacy format (no sentinel).
                content_part, log_part = markdown_content.split(LOG_SEPARATOR, 1)
                # Strip the fenced code block wrapper written by write_prompt_file().
                # Use rfind() for the closing fence instead of endswith() so the parser
                # is robust to execution_log values that lack a trailing "\n" (e.g. a
                # whitespace-only log). write_prompt_file() writes the closing "```"
                # immediately after the log content with no intervening newline in that
                # case, so endswith("\n```") would silently fail and leave the fence
                # markers in prompt_execution_log.
                log_stripped = log_part.strip()
                if log_stripped.startswith("```\n"):
                    inner = log_stripped[4:]          # remove opening "```\n" fence marker
                    close = inner.rfind("\n```")
                    if close >= 0:
                        log_body = inner[:close].strip()
                    else:
                        # No closing fence on its own line — graceful fallback: strip
                        # any trailing backticks left over from the malformed block.
                        log_body = inner.rstrip("`").strip()
                else:
                    log_body = log_stripped
                prompt_content = content_part.strip()
                prompt_execution_log = log_body
            else:
                prompt_content = markdown_content
                prompt_execution_log = ""

            prompt_id = (
                file_path.stem.split("-", 1)[0]
                if "-" in file_path.stem
                else file_path.stem
            )

            # R5 — Type-safe coercion for retry_count. YAML parses "retry_count: 3"
            # correctly as int, but a hand-edited value like "retry_count: three" or
            # "retry_count: 1.5" would pass a non-int into the dataclass, causing a
            # TypeError later when retry_count < max_retries is evaluated.
            try:
                retry_count = int(metadata.get("retry_count", 0))
            except (ValueError, TypeError):
                retry_count = 0

            prompt = QueuedPrompt(
                id=prompt_id,
                content=prompt_content,
                execution_log=prompt_execution_log,
                working_directory=metadata.get("working_directory", "."),
                priority=metadata.get("priority", 0),
                context_files=metadata.get("context_files", []),
                max_retries=metadata.get("max_retries", 3),
                retry_count=retry_count,
                estimated_tokens=metadata.get("estimated_tokens"),
                # R5 — Restore created_at from YAML; fall back to filesystem ctime.
                # Using ctime alone causes created_at to drift when files are copied or
                # their timestamps change. The YAML value is the authoritative source.
                created_at=(
                    QueueStorage._parse_optional_datetime(metadata.get("created_at"))
                    or datetime.fromtimestamp(file_path.stat().st_ctime)
                ),
                last_executed=QueueStorage._parse_optional_datetime(
                    metadata.get("last_executed")
                ),
                rate_limited_at=QueueStorage._parse_optional_datetime(
                    metadata.get("rate_limited_at")
                ),
                reset_time=QueueStorage._parse_optional_datetime(
                    metadata.get("reset_time")
                ),
            )

            return prompt

        except Exception as e:
            print(f"Error parsing prompt file {file_path}: {e}")
            return None

    @staticmethod
    def write_prompt_file(prompt: QueuedPrompt, file_path: Path) -> bool:
        """Write a QueuedPrompt to a markdown file."""
        try:
            metadata = {
                "priority": prompt.priority,
                "working_directory": prompt.working_directory,
                "max_retries": prompt.max_retries,
                "created_at": prompt.created_at.isoformat(),
                "status": prompt.status.value,
                "retry_count": prompt.retry_count,
            }

            if prompt.context_files:
                metadata["context_files"] = prompt.context_files
            if prompt.estimated_tokens:
                metadata["estimated_tokens"] = prompt.estimated_tokens
            if prompt.last_executed:
                metadata["last_executed"] = prompt.last_executed.isoformat()
            if prompt.rate_limited_at:
                metadata["rate_limited_at"] = prompt.rate_limited_at.isoformat()
            if prompt.reset_time:
                metadata["reset_time"] = prompt.reset_time.isoformat()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("---\n")
                yaml.dump(metadata, f, default_flow_style=False)
                f.write("---\n\n")
                f.write(prompt.content)

                if prompt.execution_log:
                    f.write("\n\n## Execution Log\n\n")
                    f.write("```\n")
                    f.write(prompt.execution_log)
                    f.write("```\n")

        except Exception as e:
            print(f"Error writing prompt file {file_path}: {e}")
            return False

        # SC3 — chmod is OUTSIDE the outer try — the file write already succeeded.
        # A filesystem that doesn't support permission bits (FAT, some NFS mounts)
        # must not cause write_prompt_file() to return False.
        try:
            file_path.chmod(0o600)
        except Exception as chmod_err:
            print(
                f"Warning: could not set permissions on {file_path}: {chmod_err}",
                file=sys.stderr,
            )
        return True   # write succeeded regardless of chmod outcome

    @staticmethod
    def get_base_filename(prompt: QueuedPrompt) -> str:
        """Get the base filename for a prompt (id and sanitized title, no status suffix)."""
        # R2 — Guard against empty content to prevent filenames like "{id}-.md".
        sanitized_title = QueueStorage._sanitize_filename_static(prompt.content[:50]) or "no-title"
        return f"{prompt.id}-{sanitized_title}.md"


class QueueStorage:
    """Manages queue storage using markdown files and JSON state."""

    def __init__(self, storage_dir: str = "~/.claude-queue"):
        self.base_dir = Path(storage_dir).expanduser()
        self.queue_dir = self.base_dir / "queue"
        self.completed_dir = self.base_dir / "completed"
        self.failed_dir = self.base_dir / "failed"
        self.bank_dir = self.base_dir / "bank"
        self.state_file = self.base_dir / "queue-state.json"

        # SC3 — Use unconditional chmod() after each mkdir() to ensure both new and
        # existing directories are restricted, regardless of umask. mkdir(mode=...) is
        # silently ignored when exist_ok=True and the directory already exists.
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.chmod(0o700)
        for dir_path in [self.queue_dir, self.completed_dir, self.failed_dir, self.bank_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            dir_path.chmod(0o700)

        self.parser = MarkdownPromptParser()

    @staticmethod
    def _parse_optional_datetime(val) -> Optional[datetime]:
        """Parse an ISO datetime string, returning None on any error.

        Always returns a naive datetime (tzinfo stripped) so comparisons against
        datetime.now() never raise TypeError. The existing
        _extract_reset_time_from_limit_message() can produce timezone-aware
        datetimes (e.g. from ISO strings ending in Z or +00:00); those get
        written to YAML via .isoformat() and must be normalised on read-back.

        The Z-suffix substitution handles Python < 3.11, where
        datetime.fromisoformat() does not accept the trailing "Z" character.
        Python 3.11+ accepts "Z" natively, but the substitution is harmless there.
        """
        if not val:
            return None
        try:
            if isinstance(val, str) and val.endswith("Z"):
                val = val[:-1] + "+00:00"
            return datetime.fromisoformat(val).replace(tzinfo=None)
        except (ValueError, TypeError):
            return None

    def load_queue_state(self) -> QueueState:
        """Load queue state from storage."""
        state = QueueState()

        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)

                state.total_processed = data.get("total_processed", 0)
                state.failed_count = data.get("failed_count", 0)
                state.rate_limited_count = data.get("rate_limited_count", 0)

                if data.get("last_processed"):
                    state.last_processed = datetime.fromisoformat(
                        data["last_processed"]
                    )

            except Exception as e:
                print(f"Error loading queue state: {e}")

        state.prompts = self._load_prompts_from_files()

        return state

    def save_queue_state(self, state: QueueState) -> bool:
        """Save queue state to storage."""
        try:
            self._save_prompts_to_files(state.prompts)

            state_data = {
                "total_processed": state.total_processed,
                "failed_count": state.failed_count,
                "rate_limited_count": state.rate_limited_count,
                "last_processed": (
                    state.last_processed.isoformat() if state.last_processed else None
                ),
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving queue state: {e}")
            return False

    def _load_prompts_from_files(self) -> List[QueuedPrompt]:
        """Load all prompts from markdown files."""
        prompts = []
        processed_ids = set()

        for file_path in self.queue_dir.glob("*.executing.md"):
            prompt = self.parser.parse_prompt_file(file_path)
            if prompt:
                # Fix 1c — At-least-once semantics: if the process crashed while
                # executing a prompt, the .executing.md file is left on disk.
                # Re-queue the prompt so it runs again on restart rather than
                # leaving it permanently stuck in EXECUTING state.
                prompt.status = PromptStatus.QUEUED
                prompt.add_log("Recovered from interrupted execution (process restart)")
                prompts.append(prompt)
                processed_ids.add(prompt.id)

        for file_path in self.queue_dir.glob("*.rate-limited.md"):
            prompt = self.parser.parse_prompt_file(file_path)
            if prompt:
                prompt.status = PromptStatus.RATE_LIMITED
                prompts.append(prompt)
                processed_ids.add(prompt.id)

        for file_path in self.queue_dir.glob("*.md"):
            if (
                file_path.name.endswith(".executing.md")
                or file_path.name.endswith(".rate-limited.md")
                or "#" in file_path.name
            ):
                continue

            prompt = self.parser.parse_prompt_file(file_path)
            if prompt and prompt.id not in processed_ids:
                prompt.status = PromptStatus.QUEUED
                prompts.append(prompt)

        return prompts

    def _save_prompts_to_files(self, prompts: List[QueuedPrompt]) -> None:
        """Save prompts to appropriate directories based on status."""
        for prompt in prompts:
            self._save_single_prompt(prompt)

    def _save_single_prompt(self, prompt: QueuedPrompt) -> bool:
        """Save a single prompt to the appropriate location."""
        try:
            base_filename = MarkdownPromptParser.get_base_filename(prompt)
            cross_dir = False  # will be set True for moves out of queue_dir
            if prompt.status == PromptStatus.COMPLETED:
                target_dir = self.completed_dir
                cross_dir = True
            elif prompt.status == PromptStatus.FAILED:
                target_dir = self.failed_dir
                cross_dir = True
            elif prompt.status == PromptStatus.CANCELLED:
                target_dir = self.failed_dir
                base_filename = f"{prompt.id}-cancelled.md"
                cross_dir = True
            elif prompt.status == PromptStatus.EXECUTING:
                target_dir = self.queue_dir
                base_filename = base_filename.replace(".md", ".executing.md")
                self._remove_prompt_files(prompt.id, self.queue_dir)
            elif prompt.status == PromptStatus.RATE_LIMITED:
                target_dir = self.queue_dir
                base_filename = base_filename.replace(".md", ".rate-limited.md")
                self._remove_prompt_files(prompt.id, self.queue_dir)
            else:  # QUEUED
                target_dir = self.queue_dir
                # Fix 1b — Remove any stale .executing.md or .rate-limited.md file
                # left over from a previous run that crashed before transitioning.
                # Without this, the orphaned file is picked up on reload as EXECUTING,
                # causing the retried QUEUED file to be ignored.
                self._remove_prompt_files(prompt.id, self.queue_dir)
            file_path = target_dir / base_filename
            success = self.parser.write_prompt_file(prompt, file_path)
            # FT-007 — For cross-directory moves (COMPLETED/FAILED/CANCELLED) delete
            # the queue-dir source file AFTER the destination write succeeds.
            # Deleting before writing risks permanent data loss if the write fails.
            # At-least-once caveat: a crash between write-success and unlink leaves
            # the prompt in both directories; load_queue_state will re-enqueue it.
            if cross_dir and success:
                self._remove_prompt_files(prompt.id, self.queue_dir)
            return success
        except Exception as e:
            print(f"Error saving prompt {prompt.id}: {e}")
            return False

    def _remove_prompt_files(self, prompt_id: str, directory: Path) -> None:
        """Remove all files for a prompt ID from a directory, including any status suffixes."""
        # E2 — {prompt_id}-*.md matches every variant the storage layer can write:
        # {id}-title.md, {id}-title.executing.md, {id}-title.rate-limited.md, and
        # the hard-coded {id}-cancelled.md. The `-` separator is mandatory:
        # get_base_filename() always produces `{id}-{title}.md` (falling back to
        # `{id}-no-title.md`), so no prompt file is ever written without a `-`
        # after the ID. Using `{id}*.md` (no separator) would collide with any
        # prompt whose ID is a prefix of another (e.g. removing "abc1" would also
        # delete "abc12345-task.md").
        for file_path in directory.glob(f"{prompt_id}-*.md"):
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

    @staticmethod
    def _sanitize_filename_static(text: str) -> str:
        """Sanitize text for use in filename (static version for use in parser)."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, "-")

        text = re.sub(r"[-\s]+", "-", text)
        text = text.strip("-")
        return text[:50]

    def add_prompt_from_markdown(self, file_path: Path) -> Optional[QueuedPrompt]:
        """Add a prompt from an existing markdown file."""
        prompt = self.parser.parse_prompt_file(file_path)
        if prompt:
            if file_path.parent != self.queue_dir:
                new_path = self.queue_dir / file_path.name
                shutil.move(str(file_path), str(new_path))

            prompt.status = PromptStatus.QUEUED
            return prompt
        return None

    def create_prompt_template(self, filename: str, priority: int = 0) -> Path:
        """Create a prompt template file."""
        template_content = f"""---
priority: {priority}
working_directory: .
context_files: []
max_retries: 3
estimated_tokens: null
---

# Prompt Title

Write your prompt here...

## Context
Any additional context or requirements...

## Expected Output
What should be delivered...
"""

        # FT-042 — Sanitize the caller-supplied filename before constructing the path.
        # Path(filename).name strips any leading directory components (e.g. "../../x"
        # becomes "x"), preventing writes outside queue_dir.
        # _sanitize_filename_static then removes invalid filesystem characters.
        safe_name = QueueStorage._sanitize_filename_static(Path(filename).name)
        if not safe_name:
            raise ValueError(f"Invalid template filename: {filename!r}")
        file_path = self.queue_dir / f"{safe_name}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(template_content)

        # SC3 — Restrict template file permissions.
        try:
            file_path.chmod(0o600)
        except Exception as chmod_err:
            print(
                f"Warning: could not set permissions on {file_path}: {chmod_err}",
                file=sys.stderr,
            )

        return file_path

    def save_prompt_to_bank(self, template_name: str, priority: int = 0) -> Path:
        """Save a prompt template to the bank directory."""
        # FT-043 — Sanitize before building the f-string; safe_name is referenced
        # inside template_content below so it must be computed first.
        # Path(template_name).name strips directory traversal components;
        # _sanitize_filename_static removes invalid filesystem characters.
        safe_name = QueueStorage._sanitize_filename_static(Path(template_name).name)
        if not safe_name:
            raise ValueError(f"Invalid bank template name: {template_name!r}")

        template_content = f"""---
priority: {priority}
working_directory: .
context_files: []
max_retries: 3
estimated_tokens: null
---

# {safe_name.replace('-', ' ').replace('_', ' ').title()}

Write your prompt here...

## Context
Any additional context or requirements...

## Expected Output
What should be delivered...
"""

        file_path = self.bank_dir / f"{safe_name}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(template_content)

        # SC3 — Restrict bank template file permissions.
        try:
            file_path.chmod(0o600)
        except Exception as chmod_err:
            print(
                f"Warning: could not set permissions on {file_path}: {chmod_err}",
                file=sys.stderr,
            )

        return file_path

    def list_bank_templates(self) -> List[dict]:
        """List all templates in the bank directory."""
        templates = []

        for file_path in self.bank_dir.glob("*.md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse frontmatter to get metadata
                metadata = {}
                if content.startswith("---\n"):
                    parts = content.split("---\n", 2)
                    if len(parts) >= 3:
                        try:
                            metadata = yaml.safe_load(parts[1]) or {}
                        except yaml.YAMLError:
                            pass

                # Extract title from content or use filename
                title = file_path.stem.replace('-', ' ').replace('_', ' ').title()
                if content.startswith("---\n"):
                    parts = content.split("---\n", 2)
                    if len(parts) >= 3:
                        lines = parts[2].strip().split('\n')
                        for line in lines:
                            if line.startswith('# '):
                                title = line[2:].strip()
                                break

                templates.append({
                    'name': file_path.stem,
                    'title': title,
                    'priority': metadata.get('priority', 0),
                    'working_directory': metadata.get('working_directory', '.'),
                    'estimated_tokens': metadata.get('estimated_tokens'),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                })

            except Exception as e:
                print(f"Error reading template {file_path}: {e}")
                continue

        return sorted(templates, key=lambda x: x['name'])

    def use_bank_template(self, template_name: str) -> Optional[QueuedPrompt]:
        """Copy a template from bank to queue and return as QueuedPrompt."""
        bank_file = self.bank_dir / f"{template_name}.md"

        if not bank_file.exists():
            return None

        try:
            # Parse the bank template
            template_prompt = self.parser.parse_prompt_file(bank_file)
            if not template_prompt:
                return None

            # Generate new ID for the queue
            import uuid
            new_id = str(uuid.uuid4())[:8]
            template_prompt.id = new_id
            template_prompt.status = PromptStatus.QUEUED
            template_prompt.created_at = datetime.now()
            template_prompt.retry_count = 0
            template_prompt.execution_log = ""
            template_prompt.last_executed = None
            template_prompt.rate_limited_at = None
            template_prompt.reset_time = None

            return template_prompt

        except Exception as e:
            print(f"Error using bank template {template_name}: {e}")
            return None

    def delete_bank_template(self, template_name: str) -> bool:
        """Delete a template from the bank."""
        bank_file = self.bank_dir / f"{template_name}.md"

        if not bank_file.exists():
            return False

        try:
            bank_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting bank template {template_name}: {e}")
            return False

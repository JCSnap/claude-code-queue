"""
Queue storage system with markdown support.
"""

import json
import re
import shutil
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
                except yaml.YAMLError:
                    metadata = {}

            prompt_id = (
                file_path.stem.split("-", 1)[0]
                if "-" in file_path.stem
                else file_path.stem
            )

            prompt = QueuedPrompt(
                id=prompt_id,
                content=markdown_content,
                working_directory=metadata.get("working_directory", "."),
                priority=metadata.get("priority", 0),
                context_files=metadata.get("context_files", []),
                max_retries=metadata.get("max_retries", 3),
                estimated_tokens=metadata.get("estimated_tokens"),
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
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

            return True

        except Exception as e:
            print(f"Error writing prompt file {file_path}: {e}")
            return False

    @staticmethod
    def get_base_filename(prompt: QueuedPrompt) -> str:
        """Get the base filename for a prompt (id and sanitized title, no status suffix)."""
        sanitized_title = QueueStorage._sanitize_filename_static(prompt.content[:50])
        return f"{prompt.id}-{sanitized_title}.md"


class QueueStorage:
    """Manages queue storage using markdown files and JSON state."""

    def __init__(self, base_dir: str = "~/.claude-queue"):
        self.base_dir = Path(base_dir).expanduser()
        self.queue_dir = self.base_dir / "queue"
        self.completed_dir = self.base_dir / "completed"
        self.failed_dir = self.base_dir / "failed"
        self.bank_dir = self.base_dir / "bank"
        self.state_file = self.base_dir / "queue-state.json"

        for dir_path in [self.queue_dir, self.completed_dir, self.failed_dir, self.bank_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.parser = MarkdownPromptParser()

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
                prompt.status = PromptStatus.QUEUED
                prompt.add_log("Recovered: execution was interrupted. Re-queuing.")
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
            if prompt.status == PromptStatus.COMPLETED:
                target_dir = self.completed_dir
                self._remove_prompt_files(prompt.id, self.queue_dir)
            elif prompt.status == PromptStatus.FAILED:
                target_dir = self.failed_dir
                self._remove_prompt_files(prompt.id, self.queue_dir)
            elif prompt.status == PromptStatus.CANCELLED:
                target_dir = self.failed_dir
                base_filename = f"{prompt.id}-cancelled.md"
                self._remove_prompt_files(prompt.id, self.queue_dir)
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
                self._remove_prompt_files(prompt.id, self.queue_dir)
            file_path = target_dir / base_filename
            return self.parser.write_prompt_file(prompt, file_path)
        except Exception as e:
            print(f"Error saving prompt {prompt.id}: {e}")
            return False

    def _remove_prompt_files(self, prompt_id: str, directory: Path) -> None:
        """Remove all files for a prompt ID from a directory, including any status suffixes."""
        patterns = [
            f"{prompt_id}.md",
            f"{prompt_id}*.md",
        ]
        for pattern in patterns:
            for file_path in directory.glob(pattern):
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
        for file_path in directory.glob(f"{prompt_id}-#*.md"):
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Error removing processed file {file_path}: {e}")

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

        file_path = self.queue_dir / f"{filename}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(template_content)

        return file_path

    def save_prompt_to_bank(self, template_name: str, priority: int = 0) -> Path:
        """Save a prompt template to the bank directory."""
        template_content = f"""---
priority: {priority}
working_directory: .
context_files: []
max_retries: 3
estimated_tokens: null
---

# {template_name.replace('-', ' ').replace('_', ' ').title()}

Write your prompt here...

## Context
Any additional context or requirements...

## Expected Output
What should be delivered...
"""
        
        file_path = self.bank_dir / f"{template_name}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(template_content)
        
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

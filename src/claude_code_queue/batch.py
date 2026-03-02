"""
Batch template generation for Claude Code Queue.

Generates multiple queue jobs from a template with {{variable}} placeholders
and a CSV/TSV data file where each row produces one job.
"""

import csv
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import QueuedPrompt, PromptStatus
from .storage import MarkdownPromptParser, QueueStorage

VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def extract_variables(text: str) -> Set[str]:
    """Find all {{variable}} names in template text."""
    return set(VARIABLE_PATTERN.findall(text))


def render_template(text: str, values: Dict[str, str]) -> str:
    """Replace {{var}} placeholders with values.

    Raises ValueError if any placeholders remain after substitution.
    """
    result = text
    for var, val in values.items():
        result = result.replace(f"{{{{{var}}}}}", val)

    remaining = extract_variables(result)
    if remaining:
        raise ValueError(f"Unreplaced template variables: {remaining}")

    return result


def read_data_file(data_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV/TSV data file.

    Returns (column_names, rows_as_dicts).
    Uses csv.Sniffer for automatic delimiter detection.
    """
    with open(data_path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel  # default to comma-delimited

        reader = csv.DictReader(f, dialect=dialect)
        columns = reader.fieldnames or []
        rows = list(reader)

    return columns, rows


def validate_batch(
    template_vars: Set[str],
    data_columns: List[str],
) -> Tuple[List[str], List[str]]:
    """Validate template variables against data columns.

    Returns (errors, warnings).
    """
    errors = []
    warnings = []

    col_set = set(data_columns)
    missing = template_vars - col_set
    if missing:
        errors.append(
            f"Template variables not found in data columns: {sorted(missing)}"
        )

    extra = col_set - template_vars
    if extra:
        warnings.append(
            f"Data columns not referenced in template: {sorted(extra)}"
        )

    return errors, warnings


def resolve_template_path(template_ref: str, bank_dir: Path) -> Path:
    """Resolve a template reference to a file path.

    Resolution order:
    1. If template_ref is an existing file path, use it directly
    2. Else look for bank/<template_ref>.md
    3. Else raise FileNotFoundError
    """
    # Check as direct file path
    direct = Path(template_ref)
    if direct.is_file():
        return direct

    # Check in bank directory
    bank_path = bank_dir / f"{template_ref}.md"
    if bank_path.is_file():
        return bank_path

    raise FileNotFoundError(
        f"Template '{template_ref}' not found as file path or in bank/ directory"
    )


def generate_batch_jobs(
    template_path: Path,
    data_path: Path,
    storage: QueueStorage,
    base_priority: Optional[int] = None,
    priority_step: int = 1,
    dry_run: bool = False,
) -> List[QueuedPrompt]:
    """Generate queue jobs from a template and data file.

    For each row in the data file:
    1. Render the template with the row's values
    2. Parse the rendered text as a prompt file
    3. Assign a fresh ID and reset execution state
    4. Optionally override priority with auto-incrementing values
    5. Save to queue/ (unless dry_run)

    Returns the list of generated QueuedPrompt objects.
    """
    template_text = template_path.read_text(encoding="utf-8")
    columns, rows = read_data_file(data_path)

    # Validate before generating
    template_vars = extract_variables(template_text)
    errors, warnings = validate_batch(template_vars, columns)
    if errors:
        raise ValueError("\n".join(errors))

    prompts = []
    parser = MarkdownPromptParser()

    for i, row in enumerate(rows):
        rendered = render_template(template_text, row)

        # Write to temp file so we can use the existing parser
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(rendered)
            tmp_path = Path(tmp.name)

        try:
            prompt = parser.parse_prompt_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not prompt:
            raise ValueError(f"Failed to parse rendered template for row {i + 1}: {row}")

        # Fresh identity for each generated job
        prompt.id = str(uuid.uuid4())[:8]
        prompt.status = PromptStatus.QUEUED
        prompt.created_at = datetime.now()
        prompt.retry_count = 0
        prompt.execution_log = ""
        prompt.last_executed = None
        prompt.rate_limited_at = None
        prompt.reset_time = None

        # Priority override
        if base_priority is not None:
            prompt.priority = base_priority + i * priority_step

        if not dry_run:
            storage._save_single_prompt(prompt)

        prompts.append(prompt)

    return prompts

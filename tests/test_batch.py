"""Tests for batch template generation."""

import csv
import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_code_queue.batch import (
    extract_variables,
    generate_batch_jobs,
    read_data_file,
    render_template,
    resolve_template_path,
    validate_batch,
)
from claude_code_queue.storage import QueueStorage


# --- extract_variables ---


class TestExtractVariables:
    def test_single_variable(self):
        assert extract_variables("Hello {{name}}") == {"name"}

    def test_multiple_variables(self):
        assert extract_variables("{{a}} and {{b}}") == {"a", "b"}

    def test_duplicate_variables(self):
        assert extract_variables("{{x}} then {{x}} again") == {"x"}

    def test_no_variables(self):
        assert extract_variables("plain text") == set()

    def test_variables_in_yaml_frontmatter(self):
        text = "---\nworking_directory: /src/{{project}}\n---\n\n{{task}}"
        assert extract_variables(text) == {"project", "task"}

    def test_underscores_in_variable_names(self):
        assert extract_variables("{{my_var_1}}") == {"my_var_1"}

    def test_empty_string(self):
        assert extract_variables("") == set()

    def test_single_braces_ignored(self):
        """Single curly braces (e.g. JSON/code) are not treated as variables."""
        assert extract_variables('{"key": {value}}') == set()

    def test_triple_braces_extracts_inner(self):
        # {{{var}}} should find "var" — the outer braces are literal
        result = extract_variables("{{{var}}}")
        assert "var" in result


# --- render_template ---


class TestRenderTemplate:
    def test_basic_substitution(self):
        result = render_template("Hello {{name}}", {"name": "world"})
        assert result == "Hello world"

    def test_multiple_variables(self):
        result = render_template("{{a}} + {{b}}", {"a": "1", "b": "2"})
        assert result == "1 + 2"

    def test_repeated_variable(self):
        result = render_template("{{x}} and {{x}}", {"x": "yes"})
        assert result == "yes and yes"

    def test_unreplaced_variable_raises(self):
        with pytest.raises(ValueError, match="Unreplaced template variables"):
            render_template("{{a}} {{b}}", {"a": "hello"})

    def test_extra_values_ignored(self):
        result = render_template("{{a}}", {"a": "1", "b": "2"})
        assert result == "1"

    def test_substitution_in_yaml(self):
        template = "---\nworking_directory: /src/{{project}}\n---\n\nDo {{task}}"
        result = render_template(template, {"project": "volk", "task": "review"})
        assert "/src/volk" in result
        assert "Do review" in result

    def test_empty_value_substitution(self):
        result = render_template("prefix-{{x}}-suffix", {"x": ""})
        assert result == "prefix--suffix"

    def test_value_with_special_chars(self):
        result = render_template("file: {{name}}", {"name": "test (1).h"})
        assert result == "file: test (1).h"


# --- read_data_file ---


class TestReadDataFile:
    def test_csv_comma(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nalice,30\nbob,25\n")
        columns, rows = read_data_file(csv_file)
        assert columns == ["name", "age"]
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"
        assert rows[1]["age"] == "25"

    def test_tsv(self, tmp_path):
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("name\tage\nalice\t30\nbob\t25\n")
        columns, rows = read_data_file(tsv_file)
        assert columns == ["name", "age"]
        assert len(rows) == 2

    def test_single_column(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("kernel\nvolk_a.h\nvolk_b.h\nvolk_c.h\n")
        columns, rows = read_data_file(csv_file)
        assert columns == ["kernel"]
        assert len(rows) == 3

    def test_empty_data(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\n")
        columns, rows = read_data_file(csv_file)
        assert columns == ["name", "age"]
        assert len(rows) == 0

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_data_file(tmp_path / "nonexistent.csv")

    def test_semicolon_delimiter(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name;age\nalice;30\nbob;25\n")
        columns, rows = read_data_file(csv_file)
        assert columns == ["name", "age"]
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"
        assert rows[1]["age"] == "25"

    def test_pipe_delimiter(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name|age\nalice|30\nbob|25\n")
        columns, rows = read_data_file(csv_file)
        assert columns == ["name", "age"]
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"

    def test_sniffer_fallback_to_comma(self, tmp_path):
        """When csv.Sniffer.sniff raises csv.Error, fall back to comma."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nalice,30\nbob,25\n")
        with patch("csv.Sniffer.sniff", side_effect=csv.Error("cannot sniff")):
            columns, rows = read_data_file(csv_file)
        assert columns == ["name", "age"]
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"


# --- validate_batch ---


class TestValidateBatch:
    def test_perfect_match(self):
        errors, warnings = validate_batch({"a", "b"}, ["a", "b"])
        assert errors == []
        assert warnings == []

    def test_missing_variable(self):
        errors, warnings = validate_batch({"a", "b"}, ["a"])
        assert len(errors) == 1
        assert "b" in errors[0]

    def test_extra_column_warning(self):
        errors, warnings = validate_batch({"a"}, ["a", "b"])
        assert errors == []
        assert len(warnings) == 1
        assert "b" in warnings[0]

    def test_both_missing_and_extra(self):
        errors, warnings = validate_batch({"a", "b"}, ["a", "c"])
        assert len(errors) == 1  # b missing
        assert len(warnings) == 1  # c extra

    def test_empty_template_vars(self):
        errors, warnings = validate_batch(set(), ["a"])
        assert errors == []
        assert len(warnings) == 1

    def test_empty_columns(self):
        errors, warnings = validate_batch({"a"}, [])
        assert len(errors) == 1
        assert warnings == []


# --- resolve_template_path ---


class TestResolveTemplatePath:
    def test_direct_file_path(self, tmp_path):
        template = tmp_path / "my-template.md"
        template.write_text("content")
        result = resolve_template_path(str(template), tmp_path / "bank")
        assert result == template

    def test_bank_name(self, tmp_path):
        bank_dir = tmp_path / "bank"
        bank_dir.mkdir()
        template = bank_dir / "review.md"
        template.write_text("content")
        result = resolve_template_path("review", bank_dir)
        assert result == template

    def test_not_found_raises(self, tmp_path):
        bank_dir = tmp_path / "bank"
        bank_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_template_path("nonexistent", bank_dir)

    def test_direct_path_preferred_over_bank(self, tmp_path):
        """If a file exists at the direct path, use it even if bank also has it."""
        bank_dir = tmp_path / "bank"
        bank_dir.mkdir()
        (bank_dir / "tmpl.md").write_text("bank version")
        direct = tmp_path / "tmpl.md"
        direct.write_text("direct version")
        result = resolve_template_path(str(direct), bank_dir)
        assert result == direct


# --- generate_batch_jobs ---


TEMPLATE_CONTENT = textwrap.dedent("""\
    ---
    priority: 5
    working_directory: /src/{{project}}
    context_files: []
    max_retries: 2
    estimated_tokens: 1000
    ---

    Review the file {{filename}} in project {{project}}.
""")


class TestGenerateBatchJobs:
    def _make_template_and_csv(self, tmp_path, template_text, csv_text):
        template_path = tmp_path / "template.md"
        template_path.write_text(template_text)
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(csv_text)
        return template_path, csv_path

    def test_basic_generation(self, tmp_path):
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,kernel_a.h\nvolk,kernel_b.h\n",
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)

        assert len(prompts) == 2
        assert "kernel_a.h" in prompts[0].content
        assert "kernel_b.h" in prompts[1].content
        assert prompts[0].working_directory == "/src/volk"
        assert prompts[1].working_directory == "/src/volk"

    def test_unique_ids(self, tmp_path):
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\na,x.h\nb,y.h\nc,z.h\n",
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)
        ids = [p.id for p in prompts]
        assert len(set(ids)) == 3  # all unique

    def test_default_priority_from_template(self, tmp_path):
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\nvolk,b.h\n",
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)
        assert prompts[0].priority == 5
        assert prompts[1].priority == 5

    def test_base_priority_override(self, tmp_path):
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\nvolk,b.h\nvolk,c.h\n",
        )

        prompts = generate_batch_jobs(
            template_path, csv_path, storage, base_priority=10, priority_step=5
        )
        assert prompts[0].priority == 10
        assert prompts[1].priority == 15
        assert prompts[2].priority == 20

    def test_dry_run_no_files_written(self, tmp_path):
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\n",
        )

        prompts = generate_batch_jobs(
            template_path, csv_path, storage, dry_run=True
        )
        assert len(prompts) == 1
        # Queue directory should be empty (no files written)
        queue_files = list(storage.queue_dir.glob("*.md"))
        assert len(queue_files) == 0

    def test_files_written_to_queue(self, tmp_path):
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\nvolk,b.h\n",
        )

        generate_batch_jobs(template_path, csv_path, storage)
        queue_files = list(storage.queue_dir.glob("*.md"))
        assert len(queue_files) == 2

    def test_missing_variable_in_csv_raises(self, tmp_path):
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project\nvolk\n",  # missing 'filename' column
        )

        with pytest.raises(ValueError, match="not found in data"):
            generate_batch_jobs(template_path, csv_path, storage)

    def test_max_retries_preserved(self, tmp_path):
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\n",
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)
        assert prompts[0].max_retries == 2

    def test_single_column_csv(self, tmp_path):
        """Template with one variable and a single-column CSV."""
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template = textwrap.dedent("""\
            ---
            priority: 0
            working_directory: .
            max_retries: 3
            ---

            Process {{item}}
        """)
        template_path, csv_path = self._make_template_and_csv(
            tmp_path, template, "item\nalpha\nbeta\ngamma\n"
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)
        assert len(prompts) == 3
        assert "alpha" in prompts[0].content
        assert "beta" in prompts[1].content
        assert "gamma" in prompts[2].content

    def test_working_directory_substitution(self, tmp_path):
        """Variables in YAML frontmatter values are substituted."""
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nmyproject,file.py\n",
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)
        assert prompts[0].working_directory == "/src/myproject"

    def test_execution_state_fully_reset(self, tmp_path):
        """Every state field is reset to a clean initial value per generated job."""
        from claude_code_queue.models import PromptStatus
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\nvolk,b.h\n",
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)

        for prompt in prompts:
            assert prompt.status == PromptStatus.QUEUED
            assert prompt.retry_count == 0
            assert prompt.execution_log == ""
            assert prompt.last_executed is None
            assert prompt.rate_limited_at is None
            assert prompt.reset_time is None

    def test_created_at_is_recent(self, tmp_path):
        """Each generated job has a fresh created_at timestamp."""
        from datetime import timedelta
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\nvolk,b.h\n",
        )

        before = datetime.now()
        prompts = generate_batch_jobs(template_path, csv_path, storage)
        after = datetime.now()

        for prompt in prompts:
            assert isinstance(prompt.created_at, datetime)
            assert before - timedelta(seconds=1) <= prompt.created_at <= after + timedelta(seconds=1)

    def test_id_is_eight_characters(self, tmp_path):
        """Each generated job receives an 8-character UUID prefix as its ID."""
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\nvolk,b.h\nvolk,c.h\n",
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)
        for prompt in prompts:
            assert len(prompt.id) == 8

    def test_empty_csv_returns_empty_list(self, tmp_path):
        """A CSV with only a header row produces no jobs and no files."""
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\n",  # header only, no data rows
        )

        prompts = generate_batch_jobs(template_path, csv_path, storage)

        assert prompts == []
        queue_files = list(storage.queue_dir.glob("*.md"))
        assert len(queue_files) == 0

    def test_base_priority_zero_is_applied(self, tmp_path):
        """base_priority=0 (falsy) overrides template priority — guards against 'if base_priority:' bug."""
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\n",
        )

        prompts = generate_batch_jobs(
            template_path, csv_path, storage, base_priority=0
        )
        assert prompts[0].priority == 0

    def test_priority_step_zero_all_same_priority(self, tmp_path):
        """priority_step=0 means every job gets exactly base_priority."""
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename\nvolk,a.h\nvolk,b.h\nvolk,c.h\n",
        )

        prompts = generate_batch_jobs(
            template_path, csv_path, storage, base_priority=10, priority_step=0
        )
        assert all(p.priority == 10 for p in prompts)

    def test_dry_run_with_extra_columns_still_generates(self, tmp_path):
        """Extra CSV columns are a warning, not an error; generation still proceeds."""
        storage = QueueStorage(storage_dir=str(tmp_path / "storage"))
        template_path, csv_path = self._make_template_and_csv(
            tmp_path,
            TEMPLATE_CONTENT,
            "project,filename,extra_col\nvolk,a.h,ignored\n",
        )

        prompts = generate_batch_jobs(
            template_path, csv_path, storage, dry_run=True
        )
        assert len(prompts) == 1
        assert "a.h" in prompts[0].content

#!/usr/bin/env python3
"""
Claude Code Queue - Main CLI entry point.

A tool to queue Claude Code prompts and automatically execute them when token limits reset.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from .batch import (
    extract_variables,
    generate_batch_jobs,
    read_data_file,
    resolve_template_path,
    validate_batch,
)
from .queue_manager import QueueManager
from .storage import QueueStorage
from .models import QueuedPrompt, PromptStatus


def main():
    parser = argparse.ArgumentParser(
        description="Claude Code Queue - Queue prompts and execute when limits reset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the queue processor
  python -m claude_code_queue.cli start

  # Add a quick prompt
  python -m claude_code_queue.cli add "Fix the authentication bug" --priority 1

  # Create a template for detailed prompt
  python -m claude_code_queue.cli template my-feature --priority 2

  # Launch interactive prompt box
  python -m claude_code_queue.cli prompt-box

  # Save a reusable template to bank
  python -m claude_code_queue.cli bank save update-docs --priority 1

  # List templates in bank
  python -m claude_code_queue.cli bank list

  # Use a template from bank (adds to queue)
  python -m claude_code_queue.cli bank use update-docs

  # Generate batch jobs from template + CSV
  python -m claude_code_queue.cli batch generate my-template --data entries.csv

  # Preview batch generation (dry run)
  python -m claude_code_queue.cli batch generate my-template --data entries.csv --dry-run

  # List variables in a batch template
  python -m claude_code_queue.cli batch variables my-template

  # Check queue status
  python -m claude_code_queue.cli status

  # Cancel a prompt
  python -m claude_code_queue.cli cancel abc123

  # Test Claude Code connection
  python -m claude_code_queue.cli test

  # Install the Claude Code /queue skill
  python -m claude_code_queue.cli install-skill
        """,
    )

    parser.add_argument(
        "--storage-dir",
        default="~/.claude-queue",
        help="Storage directory for queue data (default: ~/.claude-queue)",
    )

    parser.add_argument(
        "--claude-command",
        default="claude",
        help="Claude Code CLI command (default: claude)",
    )

    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Command timeout in seconds (default: 3600)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    start_parser = subparsers.add_parser("start", help="Start the queue processor")
    start_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    # SC1 — Allow operators to opt out of the global permissions bypass.
    # The flag is on the 'start' subparser only; it is meaningless for subcommands
    # that never invoke claude. Evaluating args.no_skip_permissions outside the
    # 'start' branch would risk AttributeError.
    start_parser.add_argument(
        "--no-skip-permissions",
        action="store_true",
        default=False,
        help="Do not pass --dangerously-skip-permissions to claude (requires confirmation dialogs)",
    )

    add_parser = subparsers.add_parser("add", help="Add a prompt to the queue")
    add_parser.add_argument("prompt", help="The prompt text")
    add_parser.add_argument(
        "--priority",
        "-p",
        type=int,
        default=0,
        help="Priority (lower = higher priority)",
    )
    add_parser.add_argument(
        "--working-dir", "-d", default=".", help="Working directory"
    )
    add_parser.add_argument(
        "--context-files", "-f", nargs="*", default=[], help="Context files to include"
    )
    add_parser.add_argument(
        "--max-retries", "-r", type=int, default=3, help="Maximum retry attempts"
    )
    add_parser.add_argument(
        "--estimated-tokens", "-t", type=int, help="Estimated token usage"
    )

    template_parser = subparsers.add_parser(
        "template", help="Create a prompt template file"
    )
    template_parser.add_argument(
        "filename", help="Template filename (without .md extension)"
    )
    template_parser.add_argument(
        "--priority", "-p", type=int, default=0, help="Default priority"
    )

    status_parser = subparsers.add_parser("status", help="Show queue status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    status_parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed prompt info"
    )

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a prompt")
    cancel_parser.add_argument("prompt_id", help="Prompt ID to cancel")

    list_parser = subparsers.add_parser("list", help="List prompts")
    list_parser.add_argument(
        "--status", choices=[s.value for s in PromptStatus], help="Filter by status"
    )
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    test_parser = subparsers.add_parser("test", help="Test Claude Code connection")

    # Bank subcommands
    bank_parser = subparsers.add_parser("bank", help="Manage prompt templates bank")
    bank_subparsers = bank_parser.add_subparsers(dest="bank_command", help="Bank operations")

    bank_save_parser = bank_subparsers.add_parser("save", help="Save a template to bank")
    bank_save_parser.add_argument("template_name", help="Template name for bank")
    bank_save_parser.add_argument(
        "--priority", "-p", type=int, default=0, help="Default priority"
    )

    bank_list_parser = bank_subparsers.add_parser("list", help="List templates in bank")
    bank_list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    bank_use_parser = bank_subparsers.add_parser("use", help="Use template from bank")
    bank_use_parser.add_argument("template_name", help="Template name to use")

    bank_delete_parser = bank_subparsers.add_parser("delete", help="Delete template from bank")
    bank_delete_parser.add_argument("template_name", help="Template name to delete")

    # Batch subcommands
    batch_parser = subparsers.add_parser(
        "batch", help="Generate jobs from a template and data file"
    )
    batch_subparsers = batch_parser.add_subparsers(
        dest="batch_command", help="Batch operations"
    )

    batch_generate_parser = batch_subparsers.add_parser(
        "generate", help="Generate queue jobs from template + data"
    )
    batch_generate_parser.add_argument(
        "template", help="Bank template name or path to template file"
    )
    batch_generate_parser.add_argument(
        "--data", "-d", required=True, help="Path to CSV/TSV data file"
    )
    batch_generate_parser.add_argument(
        "--base-priority", type=int, default=None,
        help="Starting priority (auto-increments per row)",
    )
    batch_generate_parser.add_argument(
        "--priority-step", type=int, default=1,
        help="Priority increment per row (default: 1)",
    )
    batch_generate_parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be generated without creating jobs",
    )

    batch_validate_parser = batch_subparsers.add_parser(
        "validate", help="Validate template variables against data columns"
    )
    batch_validate_parser.add_argument(
        "template", help="Bank template name or path to template file"
    )
    batch_validate_parser.add_argument(
        "--data", "-d", required=True, help="Path to CSV/TSV data file"
    )

    batch_variables_parser = batch_subparsers.add_parser(
        "variables", help="List template variables ({{placeholders}})"
    )
    batch_variables_parser.add_argument(
        "template", help="Bank template name or path to template file"
    )

    # Install skill subcommand
    install_skill_parser = subparsers.add_parser(
        "install-skill", help="Install the Claude Code skill to ~/.claude/skills/"
    )
    install_skill_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing skill file"
    )

    # Cleanup subcommand
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Remove rate-limit artifacts from ~/.claude/"
    )
    cleanup_parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be deleted without acting",
    )

    # Prompt box subcommand
    prompt_box_parser = subparsers.add_parser(
        "prompt-box", help="Launch the interactive prompt box CLI", add_help=False
    )
    prompt_box_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to prompt-box")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        # SC1+E3 — Dispatch without constructing a shared QueueManager upfront.
        # Constructing QueueManager requires a live claude binary (_verify_claude_available).
        # Commands that never invoke claude use QueueStorage directly, avoiding a
        # spurious PATH dependency on the claude binary for everyday queue operations.
        # cmd_start and cmd_test construct QueueManager internally.
        if args.command == "start":
            return cmd_start(args)
        elif args.command == "test":
            return cmd_test(args)
        elif args.command == "add":
            return cmd_add(args)
        elif args.command == "template":
            return cmd_template(args)
        elif args.command == "status":
            return cmd_status(args)
        elif args.command == "cancel":
            return cmd_cancel(args)
        elif args.command == "list":
            return cmd_list(args)
        elif args.command == "bank":
            return cmd_bank(args)
        elif args.command == "batch":
            return cmd_batch(args)
        elif args.command == "install-skill":
            return cmd_install_skill(args)
        elif args.command == "cleanup":
            return cmd_cleanup(args)
        elif args.command == "prompt-box":
            return cmd_prompt_box(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_start(args) -> int:
    """Start the queue processor."""
    manager = QueueManager(
        storage_dir=args.storage_dir,
        claude_command=args.claude_command,
        check_interval=args.check_interval,
        timeout=args.timeout,
        skip_permissions=not args.no_skip_permissions,
    )

    def status_callback(state):
        if args.verbose:
            stats = state.get_stats()
            print(f"Queue status: {stats['status_counts']}")

    manager.start(callback=status_callback if args.verbose else None)
    return 0


def cmd_add(args) -> int:
    """Add a prompt to the queue."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    prompt = QueuedPrompt(
        content=args.prompt,
        working_directory=args.working_dir,
        priority=args.priority,
        context_files=args.context_files,
        max_retries=args.max_retries,
        estimated_tokens=args.estimated_tokens,
    )
    # Use _save_single_prompt directly rather than load_queue_state() +
    # save_queue_state(). Loading the full queue state just to append one file
    # is unnecessary: the daemon reloads all .md files on every tick anyway,
    # so writing the file directly is sufficient. queue-state.json stores only
    # aggregate counters (total_processed, failed_count) which are not affected
    # by adding a new QUEUED prompt.
    success = storage._save_single_prompt(prompt)
    if success:
        print(f"✓ Added prompt {prompt.id} to queue")
    return 0 if success else 1


def cmd_template(args) -> int:
    """Create a prompt template file."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    file_path = storage.create_prompt_template(args.filename, args.priority)
    print(f"Created template: {file_path}")
    print("Edit the file and it will be automatically picked up by the queue processor")
    return 0


def cmd_status(args) -> int:
    """Show queue status."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    state = storage.load_queue_state()
    stats = state.get_stats()

    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    print("Claude Code Queue Status")
    print("=" * 40)
    print(f"Total prompts: {stats['total_prompts']}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Failed count: {stats['failed_count']}")
    print(f"Rate limited count: {stats['rate_limited_count']}")

    if stats["last_processed"]:
        last_processed = datetime.fromisoformat(stats["last_processed"])
        print(f"Last processed: {last_processed.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nStatus breakdown:")
    for status, count in stats["status_counts"].items():
        if count > 0:
            print(f"  {status}: {count}")

    if stats["current_rate_limit"]["is_rate_limited"]:
        reset_time = stats["current_rate_limit"]["reset_time"]
        if reset_time:
            reset_dt = datetime.fromisoformat(reset_time)
            print(f"\nRate limited until: {reset_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.detailed and state.prompts:
        print("\nPrompts:")
        print("-" * 40)
        for prompt in sorted(state.prompts, key=lambda p: p.priority):
            status_icon = {
                PromptStatus.QUEUED: "⏳",
                PromptStatus.EXECUTING: "▶️",
                PromptStatus.COMPLETED: "✅",
                PromptStatus.FAILED: "❌",
                PromptStatus.CANCELLED: "🚫",
                PromptStatus.RATE_LIMITED: "⚠️",
            }.get(prompt.status, "❓")

            print(
                f"{status_icon} {prompt.id} (P{prompt.priority}) - {prompt.status.value}"
            )
            print(
                f"   {prompt.content[:80]}{'...' if len(prompt.content) > 80 else ''}"
            )
            if prompt.retry_count > 0:
                print(f"   Retries: {prompt.retry_count}/{prompt.max_retries}")

    return 0


def cmd_cancel(args) -> int:
    """Cancel a prompt."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    state = storage.load_queue_state()
    prompt = state.get_prompt(args.prompt_id)
    if not prompt:
        print(f"Prompt {args.prompt_id} not found")
        return 1
    if prompt.status == PromptStatus.EXECUTING:
        print(f"Cannot remove executing prompt {args.prompt_id}")
        return 1
    prompt.status = PromptStatus.CANCELLED
    prompt.add_log("Cancelled by user")
    # Use _save_single_prompt directly: save_queue_state() would rewrite every
    # prompt file in the queue (O(n) writes) just to cancel one entry.
    # _save_single_prompt() writes exactly the changed file and moves it to the
    # appropriate directory. The daemon reloads on the next tick anyway.
    success = storage._save_single_prompt(prompt)
    if success:
        print(f"✓ Cancelled prompt {args.prompt_id}")
    return 0 if success else 1


def cmd_list(args) -> int:
    """List prompts."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    state = storage.load_queue_state()
    prompts = state.prompts

    if args.status:
        status_filter = PromptStatus(args.status)
        prompts = [p for p in prompts if p.status == status_filter]

    if args.json:
        prompt_data = []
        for prompt in prompts:
            prompt_data.append(
                {
                    "id": prompt.id,
                    "content": prompt.content,
                    "status": prompt.status.value,
                    "priority": prompt.priority,
                    "working_directory": prompt.working_directory,
                    "created_at": prompt.created_at.isoformat(),
                    "retry_count": prompt.retry_count,
                    "max_retries": prompt.max_retries,
                }
            )
        print(json.dumps(prompt_data, indent=2))
    else:
        if not prompts:
            print("No prompts found")
            return 0

        print(f"Found {len(prompts)} prompts:")
        print("-" * 80)
        for prompt in sorted(prompts, key=lambda p: p.priority):
            status_icon = {
                PromptStatus.QUEUED: "⏳",
                PromptStatus.EXECUTING: "▶️",
                PromptStatus.COMPLETED: "✅",
                PromptStatus.FAILED: "❌",
                PromptStatus.CANCELLED: "🚫",
                PromptStatus.RATE_LIMITED: "⚠️",
            }.get(prompt.status, "❓")

            print(
                f"{status_icon} {prompt.id} | P{prompt.priority} | {prompt.status.value}"
            )
            print(
                f"   {prompt.content[:70]}{'...' if len(prompt.content) > 70 else ''}"
            )
            print(f"   Created: {prompt.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


def cmd_test(args) -> int:
    """Test Claude Code connection."""
    manager = QueueManager(
        storage_dir=args.storage_dir,
        claude_command=args.claude_command,
        check_interval=args.check_interval,
        timeout=args.timeout,
    )
    is_working, message = manager.claude_interface.test_connection()
    print(message)
    return 0 if is_working else 1


def cmd_bank(args) -> int:
    """Handle bank subcommands."""
    if not args.bank_command:
        print("Error: No bank operation specified")
        print("Available operations: save, list, use, delete")
        return 1

    if args.bank_command == "save":
        return cmd_bank_save(args)
    elif args.bank_command == "list":
        return cmd_bank_list(args)
    elif args.bank_command == "use":
        return cmd_bank_use(args)
    elif args.bank_command == "delete":
        return cmd_bank_delete(args)
    else:
        print(f"Unknown bank operation: {args.bank_command}")
        return 1


def cmd_bank_save(args) -> int:
    """Save a template to the bank."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    file_path = storage.save_prompt_to_bank(args.template_name, args.priority)
    print(f"✓ Created template in bank: {file_path}")
    print(f"Edit {file_path} to customize your template")
    return 0


def cmd_bank_list(args) -> int:
    """List templates in the bank."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    templates = storage.list_bank_templates()

    if args.json:
        print(json.dumps(templates, indent=2, default=str))
        return 0

    if not templates:
        print("No templates found in bank")
        return 0

    print(f"Found {len(templates)} template(s) in bank:")
    print("-" * 80)

    for template in templates:
        print(f"📄 {template['name']}")
        print(f"   Title: {template['title']}")
        print(f"   Priority: {template['priority']}")
        print(f"   Working directory: {template['working_directory']}")
        if template['estimated_tokens']:
            print(f"   Estimated tokens: {template['estimated_tokens']}")
        print(f"   Modified: {template['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    return 0


def cmd_bank_use(args) -> int:
    """Use a template from the bank."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    prompt = storage.use_bank_template(args.template_name)
    if not prompt:
        print(f"Template '{args.template_name}' not found in bank")
        return 1
    success = storage._save_single_prompt(prompt)
    if success:
        print(f"✓ Added prompt {prompt.id} from template '{args.template_name}' to queue")
    else:
        print(f"✗ Failed to save prompt from template '{args.template_name}'")
    return 0 if success else 1


def cmd_bank_delete(args) -> int:
    """Delete a template from the bank."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    success = storage.delete_bank_template(args.template_name)
    if success:
        print(f"✓ Deleted template '{args.template_name}' from bank")
    else:
        print(f"✗ Template '{args.template_name}' not found in bank")
    return 0 if success else 1


def cmd_batch(args) -> int:
    """Handle batch subcommands."""
    if not args.batch_command:
        print("Error: No batch operation specified")
        print("Available operations: generate, validate, variables")
        return 1

    if args.batch_command == "generate":
        return cmd_batch_generate(args)
    elif args.batch_command == "validate":
        return cmd_batch_validate(args)
    elif args.batch_command == "variables":
        return cmd_batch_variables(args)
    else:
        print(f"Unknown batch operation: {args.batch_command}")
        return 1


def cmd_batch_generate(args) -> int:
    """Generate queue jobs from a template and data file."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    data_path = Path(args.data)

    if not data_path.is_file():
        print(f"Data file not found: {args.data}")
        return 1

    try:
        template_path = resolve_template_path(args.template, storage.bank_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    try:
        prompts = generate_batch_jobs(
            template_path=template_path,
            data_path=data_path,
            storage=storage,
            base_priority=args.base_priority,
            priority_step=args.priority_step,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    if args.dry_run:
        print(f"Dry run: would generate {len(prompts)} job(s)")
        print("-" * 60)
        for i, prompt in enumerate(prompts):
            print(f"  [{i + 1}] P{prompt.priority} | {prompt.working_directory}")
            print(f"      {prompt.content[:70]}{'...' if len(prompt.content) > 70 else ''}")
    else:
        print(f"Generated {len(prompts)} job(s) from template '{args.template}'")
        if prompts:
            priorities = [p.priority for p in prompts]
            print(f"  Priority range: {min(priorities)}–{max(priorities)}")

    return 0


def cmd_batch_validate(args) -> int:
    """Validate template variables against data columns."""
    storage = QueueStorage(storage_dir=args.storage_dir)
    data_path = Path(args.data)

    if not data_path.is_file():
        print(f"Data file not found: {args.data}")
        return 1

    try:
        template_path = resolve_template_path(args.template, storage.bank_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    template_text = template_path.read_text(encoding="utf-8")
    template_vars = extract_variables(template_text)
    columns, rows = read_data_file(data_path)
    errors, warnings = validate_batch(template_vars, columns)

    print(f"Template: {template_path}")
    print(f"Data: {data_path} ({len(rows)} row(s), {len(columns)} column(s))")
    print(f"Variables: {sorted(template_vars) if template_vars else '(none)'}")
    print(f"Columns: {columns if columns else '(none)'}")

    if errors:
        for err in errors:
            print(f"\nError: {err}")
    if warnings:
        for warn in warnings:
            print(f"\nWarning: {warn}")

    if not errors and not warnings:
        print(f"\nValid: {len(rows)} job(s) would be generated")
    elif not errors:
        print(f"\nValid with warnings: {len(rows)} job(s) would be generated")

    return 1 if errors else 0


def cmd_batch_variables(args) -> int:
    """List template variables."""
    storage = QueueStorage(storage_dir=args.storage_dir)

    try:
        template_path = resolve_template_path(args.template, storage.bank_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    template_text = template_path.read_text(encoding="utf-8")
    variables = extract_variables(template_text)

    print(f"Template: {template_path}")
    if variables:
        print(f"Variables ({len(variables)}):")
        for var in sorted(variables):
            print(f"  {{{{{var}}}}}")
    else:
        print("No variables found in template")

    return 0


def cmd_install_skill(args) -> int:
    """Install the Claude Code skill file to ~/.claude/skills/queue/SKILL.md."""
    dest = Path.home() / ".claude" / "skills" / "queue" / "SKILL.md"
    skill_src = Path(__file__).parent / "skills" / "queue" / "SKILL.md"

    if not skill_src.exists():
        print("Error: bundled SKILL.md not found in package installation.")
        return 1

    if dest.exists() and not args.force:
        print(f"Skill already installed at {dest}")
        print("Use --force to overwrite.")
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(skill_src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Skill installed to {dest}")
    print("Restart Claude Code for the /queue skill to become available.")
    return 0


def cmd_cleanup(args) -> int:
    """Remove rate-limit artifacts from ~/.claude/.

    Primary identification: scan debug transcripts for 'rate_limit_error' in
    the content (authoritative signal).  Then delete correlated JSONL, todo,
    and telemetry files by UUID.

    This is the E3 pattern: no claude binary needed.
    """
    claude_dir = Path.home() / ".claude"
    dry_run = args.dry_run
    matched = 0
    skipped = 0
    rate_limited_uuids: List[str] = []

    # 1. Debug transcripts — primary identification via content grep.
    #    Read the full file (max ~90 KB for successful runs) since this is a
    #    one-time tool where correctness matters more than speed.
    debug_dir = claude_dir / "debug"
    if debug_dir.is_dir():
        for debug_file in debug_dir.glob("*.txt"):
            try:
                with open(debug_file, "r", errors="replace") as fh:
                    content = fh.read()
                if "rate_limit_error" in content:
                    rate_limited_uuids.append(debug_file.stem)
                    if dry_run:
                        print(f"  [dry-run] would delete {debug_file}")
                    else:
                        debug_file.unlink()
                    matched += 1
            except OSError:
                skipped += 1

    if rate_limited_uuids:
        print(f"Identified {len(rate_limited_uuids)} rate-limited session(s)")

    # 2. JSONL conversation logs — by UUID correlation
    projects_dir = claude_dir / "projects"
    if projects_dir.is_dir():
        for session_uuid in rate_limited_uuids:
            for jsonl_file in projects_dir.glob(f"*/{session_uuid}.jsonl"):
                try:
                    if dry_run:
                        print(f"  [dry-run] would delete {jsonl_file}")
                    else:
                        jsonl_file.unlink()
                    matched += 1
                except OSError:
                    skipped += 1

    # 3. Todo stubs — by UUID correlation + 2-byte size guard
    todos_dir = claude_dir / "todos"
    if todos_dir.is_dir():
        for session_uuid in rate_limited_uuids:
            todo_file = todos_dir / f"{session_uuid}-agent-{session_uuid}.json"
            try:
                st = todo_file.stat()
                if st.st_size <= 2:
                    if dry_run:
                        print(f"  [dry-run] would delete {todo_file}")
                    else:
                        todo_file.unlink()
                    matched += 1
            except OSError:
                skipped += 1

    # 4. Telemetry — by UUID correlation
    telemetry_dir = claude_dir / "telemetry"
    if telemetry_dir.is_dir():
        for session_uuid in rate_limited_uuids:
            for f in telemetry_dir.glob(f"1p_failed_events.{session_uuid}.*.json"):
                try:
                    if dry_run:
                        print(f"  [dry-run] would delete {f}")
                    else:
                        f.unlink()
                    matched += 1
                except OSError:
                    skipped += 1

    action = "Would delete" if dry_run else "Deleted"
    print(f"{action} {matched} rate-limit artifact(s)")
    if skipped:
        print(f"Skipped {skipped} file(s) due to errors")
    return 0


def cmd_prompt_box(args) -> int:
    """Launch the interactive prompt box CLI."""
    try:
        import shutil as _shutil

        binary_name = "prompt-box"
        if sys.platform == "win32":
            binary_name += ".exe"

        # Try to find the binary in PATH first
        binary_path = _shutil.which(binary_name)

        if not binary_path:
            # Fallback: check in the same directory as the Python executable
            python_bin_dir = os.path.dirname(sys.executable)
            potential_path = os.path.join(python_bin_dir, binary_name)
            if os.path.exists(potential_path):
                binary_path = potential_path

        if not binary_path or not os.path.exists(binary_path):
            print(
                "Error: prompt-box binary not found.\n"
                "This feature requires the Rust toolchain to be present at install time.\n"
                "To enable it:\n"
                "  1. Install Rust: https://rustup.rs\n"
                "  2. Reinstall: pip install --force-reinstall claude-code-queue"
            )
            return 1

        # Execute the Rust binary with all arguments
        result = subprocess.run([binary_path] + args.args, stdout=sys.stdout, stderr=sys.stderr)
        return result.returncode

    except FileNotFoundError:
        print(f"Error: Could not execute prompt-box binary")
        return 1
    except Exception as e:
        print(f"Error launching prompt-box: {e}")
        return 1


if __name__ == "__main__":
    main()

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
from typing import Dict, Optional, Set
from uuid import UUID

from .batch import (
    extract_variables,
    generate_batch_jobs,
    read_data_file,
    resolve_template_path,
    validate_batch,
)
from .queue_manager import QueueManager
from .storage import QueueStorage
from .models import (
    QueuedPrompt,
    PromptStatus,
    parse_optional_model,
    parse_optional_profile_dir,
    parse_optional_session_id,
)
from .config import PROJECT_CONFIG_FILENAME, resolve_resume_message
from .paths import claude_config_dir
from .sessions import find_session, list_sessions


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


_RATE_LIMIT_PREFIXES = (
    "Error: 429",
    "Error: Error: 429",
    "Error in non-streaming fallback: 429",
)


def _is_rate_limit_error_record(line: str) -> bool:
    """Return whether a timestamped Claude error line is an Anthropic 429."""
    timestamp, separator, record = line.strip().partition(" [ERROR] ")
    if not separator:
        return False

    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return False

    prefix = next(
        (candidate for candidate in _RATE_LIMIT_PREFIXES if record.startswith(candidate)),
        None,
    )
    if prefix is None:
        return False

    try:
        payload = json.loads(record[len(prefix):].strip())
    except json.JSONDecodeError:
        return False

    if not isinstance(payload, dict) or payload.get("type") != "error":
        return False
    error = payload.get("error")
    return isinstance(error, dict) and error.get("type") == "rate_limit_error"


def _contains_rate_limit_error_record(content: str) -> bool:
    """Return whether a Claude debug transcript contains a real 429 record."""
    return any(_is_rate_limit_error_record(line) for line in content.splitlines())


def _is_non_symlink_directory(path: Path) -> bool:
    """Return whether path is a directory rather than a directory symlink."""
    return path.is_dir() and not path.is_symlink()


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

  # Install all bundled Claude Code skills
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

    sessions_parser = subparsers.add_parser(
        "sessions",
        help="List Claude Code sessions with their ids and titles",
        description=(
            "List Claude Code conversations, newest first, with the session id "
            "`resume-session` needs. Defaults to sessions started in the current "
            "directory."
        ),
    )
    sessions_parser.add_argument(
        "project",
        nargs="?",
        metavar="PROJECT",
        help="Directory to list sessions for (default: the current directory)",
    )
    sessions_parser.add_argument(
        "--all", "-a", action="store_true",
        help="Every project, not just one directory",
    )
    sessions_parser.add_argument(
        "--search", "-s", help="Only sessions whose title contains this text",
    )
    sessions_parser.add_argument(
        "--limit", "-n", type=_nonnegative_int, default=20,
        help="Maximum sessions to show (default: 20, 0 for no limit)",
    )
    sessions_parser.add_argument(
        "--json", action="store_true", help="Output as JSON",
    )

    resume_parser = subparsers.add_parser(
        "resume-session",
        help="Queue a continuation of an existing Claude Code session",
        description=(
            "Queue a job that continues an existing Claude Code conversation once "
            "the usage limit resets. With no SESSION_ID, the session Claude Code "
            "exported as $CLAUDE_CODE_SESSION_ID is used, so this can be run from "
            "inside the session that hit the limit."
        ),
    )
    resume_parser.add_argument(
        "session_id",
        nargs="?",
        metavar="SESSION_ID",
        help="Session to continue (default: $CLAUDE_CODE_SESSION_ID)",
    )
    resume_parser.add_argument(
        "--message",
        "-m",
        help="Message to send when continuing (default: the configured resume message)",
    )
    resume_parser.add_argument(
        "--priority", "-p", type=int, default=0,
        help="Priority (lower = higher priority)",
    )
    resume_parser.add_argument(
        "--working-dir", "-d",
        help="Directory the continuation runs in (default: the session's own)",
    )
    resume_parser.add_argument(
        "--profile", metavar="DIR",
        help="Claude Code profile (config directory) to bill (default: the active one)",
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
        "--profile", metavar="DIR",
        help="Claude Code profile (config directory) to bill (default: the active one)",
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
    add_parser.add_argument(
        "--model",
        "-m",
        type=parse_optional_model,
        default=None,
        help="Claude model ID (e.g. claude-haiku-4-5-20251001)",
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
        "install-skill",
        help="Install bundled Claude Code skills into the active profile",
    )
    install_skill_parser.add_argument(
        "skill_name",
        nargs="?",
        help="Skill to install (default: all bundled skills)",
    )
    install_skill_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing skill file"
    )

    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Remove rate-limit artifacts from the active Claude profile"
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
        elif args.command == "sessions":
            return cmd_sessions(args)
        elif args.command == "resume-session":
            return cmd_resume_session(args)
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

    started = manager.start(callback=status_callback if args.verbose else None)
    return 0 if started else 1


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
        model=args.model,
        claude_config_dir=_resolve_profile(args.profile),
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
        print(f"  Profile: {prompt.claude_config_dir}")
    return 0 if success else 1


def _resolve_profile(explicit: Optional[str] = None) -> str:
    """Decide which Claude Code profile a queued prompt should bill to.

    Recorded at queue time rather than left to the processor. Each config
    directory holds its own credentials, so the profile decides which account
    pays; a processor started under a different one would otherwise spend the
    wrong account's budget without saying so.
    """
    chosen = Path(parse_optional_profile_dir(explicit) or str(claude_config_dir()))
    if not chosen.is_dir():
        raise ValueError(f"profile directory {chosen} does not exist")
    return str(chosen)


def _format_age(moment: datetime) -> str:
    """Render how long ago *moment* was, compactly enough for a table column.

    Rolls over on the units people actually read by — minutes at an hour, hours
    at a day — and gives up on relative time after a year, where a date says more.
    """
    seconds = max(0, int((datetime.now() - moment).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    days = seconds // 86400
    if days < 365:
        return f"{days}d ago"
    return moment.strftime("%Y-%m-%d")


def cmd_sessions(args) -> int:
    """List Claude Code sessions so a session id can be found without guesswork.

    Scoped to the current directory by default: the question is almost always
    "which of my sessions in this project was that one?".
    """
    if args.all and args.project:
        print(
            f"Error: --all lists every project; drop it to scope to {args.project}.",
            file=sys.stderr,
        )
        return 1

    project = None if args.all else (args.project or os.getcwd())
    if args.project and not Path(args.project).expanduser().is_dir():
        # A warning, not an error: a project can be deleted while its transcripts
        # remain, and listing those is a legitimate thing to want.
        print(f"Warning: {args.project} is not a directory.", file=sys.stderr)

    limit = None if args.limit == 0 else args.limit
    found = list_sessions(project_dir=project, search=args.search, limit=limit)

    if args.json:
        print(json.dumps([s.to_dict() for s in found], indent=2, ensure_ascii=False))
        return 0

    if not found:
        where = "any project" if args.all else f"{project}"
        print(f"No Claude Code sessions found for {where}.")
        if not args.all:
            print("Use --all to list sessions from every project.")
        return 0

    ages = [_format_age(s.last_active) for s in found]
    age_width = max([len("LAST ACTIVE")] + [len(a) for a in ages])
    print(f"{'SESSION ID':36}  {'LAST ACTIVE'.ljust(age_width)}  TITLE")
    for session, age in zip(found, ages):
        title = session.title
        if len(title) > 68:
            title = title[:67] + "…"
        print(f"{session.session_id:36}  {age.ljust(age_width)}  {title}")

    if args.all:
        print()
        print("Project directories vary; pass one to narrow the list: "
              "claude-queue sessions DIR")
    print()
    print("Continue one when the limit resets:")
    print(f"  claude-queue resume-session {found[0].session_id}")
    return 0


def cmd_resume_session(args) -> int:
    """Queue a continuation of an existing Claude Code session.

    Claude Code exports the running session's id as $CLAUDE_CODE_SESSION_ID, so
    this can be run from inside the session that hit the usage limit: the queue
    picks the conversation back up at reset instead of the user having to
    remember to.

    The continuation runs non-interactively via ``claude --print --resume``. The
    conversation is a normal session afterwards, so it can still be reopened with
    ``claude --resume <id>`` with the queue's work already in its history.
    """
    session_from_environment = args.session_id is None
    session_id = args.session_id or os.environ.get("CLAUDE_CODE_SESSION_ID")
    if not session_id:
        print(
            "Error: no session id given and $CLAUDE_CODE_SESSION_ID is not set.\n"
            "Pass one explicitly: claude-queue resume-session <session-id>",
            file=sys.stderr,
        )
        return 1

    # Validate now rather than letting the CLI reject it at execution time, hours
    # later, after the reset the user was waiting for.
    try:
        session_id = parse_optional_session_id(session_id)
    except ValueError:
        print(
            f"Error: {session_id!r} is not a valid session id (expected a UUID).",
            file=sys.stderr,
        )
        return 1

    # A session belongs to the directory it was working in. Defaulting to the
    # caller's directory instead would resume the conversation somewhere else
    # entirely — wrong files, wrong repository — which is never what "continue
    # this session" means.
    profile = _resolve_profile(args.profile)
    known = find_session(session_id, Path(profile))
    if args.working_dir is not None:
        working_dir = str(Path(args.working_dir).expanduser().resolve())
    elif known and known.project_dir:
        working_dir = known.project_dir
    elif session_from_environment:
        # Claude exports this id from the owning live conversation. Its current
        # process directory is a stronger ownership signal than a not-yet-flushed log.
        working_dir = os.getcwd()
    else:
        print(
            f"Error: no usable log for session {session_id} in profile {profile}. "
            "Pass --working-dir only if this profile owns that session.",
            file=sys.stderr,
        )
        return 1

    if known is None:
        print(
            f"Warning: no log for session {session_id} in this profile. It may "
            "belong to another CLAUDE_CONFIG_DIR; the explicit working directory "
            "will be used, but Claude Code may reject the resume.",
            file=sys.stderr,
        )

    storage = QueueStorage(storage_dir=args.storage_dir)
    prompt = QueuedPrompt(
        content=(
            f"Continue Claude Code session `{session_id}`.\n\n"
            "The queue resumes this conversation when the usage limit resets. The "
            "message it sends is the `resume_message` below, falling back to "
            f"`{PROJECT_CONFIG_FILENAME}` in the working directory, then the "
            "queue's `config.yaml`, then the built-in default."
        ),
        working_directory=working_dir,
        priority=args.priority,
        session_id=session_id,
        resume_existing_session=True,
        resume_message=args.message,
        claude_config_dir=profile,
    )

    if not storage._save_single_prompt(prompt):
        return 1

    message = resolve_resume_message(args.message, working_dir, args.storage_dir)
    print(f"✓ Queued continuation of session {session_id} as prompt {prompt.id}")
    if known is not None:
        print(f"  Session: {known.title}")
    print(f"  Working directory: {working_dir}")
    print(f"  Profile: {profile}")
    print(f"  Will send: {message}")
    return 0


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
            if prompt.model is not None:
                print(f"   Model: {prompt.model}")
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
                    "model": prompt.model,
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
            if prompt.model is not None:
                print(f"   Model: {prompt.model}")
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
        if template.get('model'):
            print(f"   Model: {template['model']}")
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
    try:
        columns, rows = read_data_file(data_path)
    except (ValueError, OSError) as e:
        print(f"Error: {e}")
        return 1
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
    """Install one or all bundled skills into the active Claude profile."""
    skills_pkg_dir = Path(__file__).parent / "skills"
    if not skills_pkg_dir.is_dir():
        print("Error: bundled skills directory not found in package installation.")
        return 1

    try:
        bundled = {
            path.name: path / "SKILL.md"
            for path in sorted(skills_pkg_dir.iterdir())
            if path.is_dir() and (path / "SKILL.md").is_file()
        }
    except OSError as e:
        print(f"Error: could not read bundled skills: {e}")
        return 1

    if not bundled:
        print("Error: no bundled skills found in package installation.")
        return 1

    skill_name = getattr(args, "skill_name", None)
    if skill_name and skill_name not in bundled:
        print(f"Error: unknown bundled skill '{skill_name}'.")
        print(f"Available skills: {', '.join(bundled)}")
        return 1

    selected = [skill_name] if skill_name else list(bundled)
    skills_dir = claude_config_dir() / "skills"
    destinations = {name: skills_dir / name / "SKILL.md" for name in selected}

    conflicts = [dest for dest in destinations.values() if dest.exists()]
    if conflicts and not args.force:
        for dest in conflicts:
            print(f"Skill already installed at {dest}")
        print("Use --force to overwrite.")
        return 1

    try:
        contents = {
            name: bundled[name].read_text(encoding="utf-8") for name in selected
        }
    except OSError as e:
        print(f"Error: could not read bundled skill: {e}")
        return 1

    try:
        originals = {
            name: dest.read_bytes() if dest.exists() else None
            for name, dest in destinations.items()
        }
    except OSError as e:
        print(f"Error: could not read installed skill: {e}")
        return 1
    written = []
    try:
        for name, dest in destinations.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            written.append(name)
            dest.write_text(contents[name], encoding="utf-8")
    except OSError as e:
        for name in reversed(written):
            dest = destinations[name]
            original = originals[name]
            try:
                if original is None:
                    dest.unlink(missing_ok=True)
                else:
                    dest.write_bytes(original)
            except OSError:
                pass
        print(f"Error: could not install bundled skills: {e}")
        return 1

    for dest in destinations.values():
        print(f"Skill installed to {dest}")
    print("Restart Claude Code for the installed skills to become available.")
    return 0


def cmd_cleanup(args) -> int:
    """Remove rate-limit artifacts from the active Claude profile.

    Identify sessions from canonical-UUID debug transcripts that contain a
    429 rate_limit_error record. Delete correlated JSONL, empty todo, telemetry,
    and empty session environment artifacts before the debug marker.
    """
    claude_dir = claude_config_dir()
    dry_run = args.dry_run
    matched = 0
    skipped = 0
    rate_limited_sessions: Dict[str, Path] = {}
    failed_sessions: Set[str] = set()

    debug_dir = claude_dir / "debug"
    if debug_dir.is_symlink():
        skipped += 1
    elif _is_non_symlink_directory(debug_dir):
        for debug_file in debug_dir.glob("*.txt"):
            try:
                session_uuid = str(UUID(debug_file.stem))
            except ValueError:
                continue
            if session_uuid != debug_file.stem:
                continue
            if debug_file.is_symlink():
                skipped += 1
                failed_sessions.add(session_uuid)
                continue

            try:
                with open(debug_file, "r", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                skipped += 1
                failed_sessions.add(session_uuid)
                continue

            has_rate_limit_error = _contains_rate_limit_error_record(content)
            if has_rate_limit_error:
                rate_limited_sessions[session_uuid] = debug_file

    if rate_limited_sessions:
        print(f"Identified {len(rate_limited_sessions)} rate-limited session(s)")

    projects_dir = claude_dir / "projects"
    if projects_dir.is_symlink():
        skipped += 1
        failed_sessions.update(rate_limited_sessions)
    elif _is_non_symlink_directory(projects_dir):
        try:
            project_dirs = []
            for path in projects_dir.iterdir():
                if path.is_symlink():
                    skipped += 1
                    failed_sessions.update(rate_limited_sessions)
                elif _is_non_symlink_directory(path):
                    project_dirs.append(path)
        except OSError:
            skipped += 1
            failed_sessions.update(rate_limited_sessions)
            project_dirs = []

        for project_dir in project_dirs:
            for session_uuid in rate_limited_sessions:
                jsonl_file = project_dir / f"{session_uuid}.jsonl"
                try:
                    if dry_run:
                        if not jsonl_file.is_symlink() and not jsonl_file.is_file():
                            continue
                        print(f"  [dry-run] would delete {jsonl_file}")
                    else:
                        jsonl_file.unlink()
                    matched += 1
                except FileNotFoundError:
                    pass
                except OSError:
                    skipped += 1
                    failed_sessions.add(session_uuid)

    todos_dir = claude_dir / "todos"
    if todos_dir.is_symlink():
        skipped += 1
        failed_sessions.update(rate_limited_sessions)
    elif _is_non_symlink_directory(todos_dir):
        for session_uuid in rate_limited_sessions:
            todo_file = todos_dir / f"{session_uuid}-agent-{session_uuid}.json"
            try:
                if todo_file.is_symlink():
                    if dry_run:
                        print(f"  [dry-run] would delete {todo_file}")
                    else:
                        todo_file.unlink()
                    matched += 1
                    continue
                st = todo_file.stat()
                if st.st_size <= 2:
                    if dry_run:
                        print(f"  [dry-run] would delete {todo_file}")
                    else:
                        todo_file.unlink()
                    matched += 1
            except FileNotFoundError:
                pass
            except OSError:
                skipped += 1
                failed_sessions.add(session_uuid)

    telemetry_dir = claude_dir / "telemetry"
    if telemetry_dir.is_symlink():
        skipped += 1
        failed_sessions.update(rate_limited_sessions)
    elif _is_non_symlink_directory(telemetry_dir):
        for session_uuid in rate_limited_sessions:
            for f in telemetry_dir.glob(f"1p_failed_events.{session_uuid}.*.json"):
                try:
                    if dry_run:
                        print(f"  [dry-run] would delete {f}")
                    else:
                        f.unlink()
                    matched += 1
                except OSError:
                    skipped += 1
                    failed_sessions.add(session_uuid)

    session_env_root = claude_dir / "session-env"
    if session_env_root.is_symlink():
        skipped += 1
        failed_sessions.update(rate_limited_sessions)
    elif _is_non_symlink_directory(session_env_root):
        for session_uuid in rate_limited_sessions:
            session_env_dir = session_env_root / session_uuid
            if session_env_dir.is_symlink():
                try:
                    if dry_run:
                        print(f"  [dry-run] would delete {session_env_dir}")
                    else:
                        session_env_dir.unlink()
                    matched += 1
                except OSError:
                    skipped += 1
                    failed_sessions.add(session_uuid)
                continue
            if not _is_non_symlink_directory(session_env_dir):
                continue
            try:
                if any(session_env_dir.iterdir()):
                    continue
                if dry_run:
                    print(f"  [dry-run] would delete {session_env_dir}")
                else:
                    session_env_dir.rmdir()
                matched += 1
            except OSError:
                skipped += 1
                failed_sessions.add(session_uuid)

    for session_uuid, debug_file in rate_limited_sessions.items():
        if session_uuid in failed_sessions:
            continue
        try:
            if dry_run:
                print(f"  [dry-run] would delete {debug_file}")
            else:
                debug_file.unlink()
            matched += 1
        except OSError:
            skipped += 1

    action = "Would delete" if dry_run else "Deleted"
    print(f"{action} {matched} rate-limit artifact(s)")
    if skipped:
        print(f"Skipped {skipped} file(s) due to errors")
    return 1 if skipped else 0


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

        if not binary_path:
            # Fallback: check the Cargo build output relative to this package
            pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            for profile in ("release", "debug"):
                potential_path = os.path.join(pkg_dir, "claude-prompt-box", "target", profile, binary_name)
                if os.path.exists(potential_path):
                    binary_path = potential_path
                    break

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
    raise SystemExit(main())

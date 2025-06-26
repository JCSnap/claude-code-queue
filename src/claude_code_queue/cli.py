#!/usr/bin/env python3
"""
Claude Code Queue - Main CLI entry point.

A tool to queue Claude Code prompts and automatically execute them when token limits reset.
"""

import argparse
import json
from datetime import datetime

from .queue_manager import QueueManager
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

  # Check queue status
  python -m claude_code_queue.cli status

  # Cancel a prompt
  python -m claude_code_queue.cli cancel abc123

  # Test Claude Code connection  
  python -m claude_code_queue.cli test
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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        manager = QueueManager(
            storage_dir=args.storage_dir,
            claude_command=args.claude_command,
            check_interval=args.check_interval,
            timeout=args.timeout,
        )

        if args.command == "start":
            return cmd_start(manager, args)
        elif args.command == "add":
            return cmd_add(manager, args)
        elif args.command == "template":
            return cmd_template(manager, args)
        elif args.command == "status":
            return cmd_status(manager, args)
        elif args.command == "cancel":
            return cmd_cancel(manager, args)
        elif args.command == "list":
            return cmd_list(manager, args)
        elif args.command == "test":
            return cmd_test(manager, args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_start(manager: QueueManager, args) -> int:
    """Start the queue processor."""

    def status_callback(state):
        if args.verbose:
            stats = state.get_stats()
            print(f"Queue status: {stats['status_counts']}")

    manager.start(callback=status_callback if args.verbose else None)
    return 0


def cmd_add(manager: QueueManager, args) -> int:
    """Add a prompt to the queue."""
    prompt = QueuedPrompt(
        content=args.prompt,
        working_directory=args.working_dir,
        priority=args.priority,
        context_files=args.context_files,
        max_retries=args.max_retries,
        estimated_tokens=args.estimated_tokens,
    )

    success = manager.add_prompt(prompt)
    return 0 if success else 1


def cmd_template(manager: QueueManager, args) -> int:
    """Create a prompt template file."""
    file_path = manager.create_prompt_template(args.filename, args.priority)
    print(f"Created template: {file_path}")
    print("Edit the file and it will be automatically picked up by the queue processor")
    return 0


def cmd_status(manager: QueueManager, args) -> int:
    """Show queue status."""
    state = manager.get_status()
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


def cmd_cancel(manager: QueueManager, args) -> int:
    """Cancel a prompt."""
    success = manager.remove_prompt(args.prompt_id)
    return 0 if success else 1


def cmd_list(manager: QueueManager, args) -> int:
    """List prompts."""
    state = manager.get_status()
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


def cmd_test(manager: QueueManager, args) -> int:
    """Test Claude Code connection."""
    is_working, message = manager.claude_interface.test_connection()
    print(message)
    return 0 if is_working else 1

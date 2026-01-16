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

  # Launch interactive prompt box
  python -m claude_code_queue.cli prompt-box

  # Save a reusable template to bank
  python -m claude_code_queue.cli bank save update-docs --priority 1

  # List templates in bank
  python -m claude_code_queue.cli bank list

  # Use a template from bank (adds to queue)
  python -m claude_code_queue.cli bank use update-docs

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
        elif args.command == "bank":
            return cmd_bank(manager, args)
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


def cmd_bank(manager: QueueManager, args) -> int:
    """Handle bank subcommands."""
    if not args.bank_command:
        print("Error: No bank operation specified")
        print("Available operations: save, list, use, delete")
        return 1

    if args.bank_command == "save":
        return cmd_bank_save(manager, args)
    elif args.bank_command == "list":
        return cmd_bank_list(manager, args)
    elif args.bank_command == "use":
        return cmd_bank_use(manager, args)
    elif args.bank_command == "delete":
        return cmd_bank_delete(manager, args)
    else:
        print(f"Unknown bank operation: {args.bank_command}")
        return 1


def cmd_bank_save(manager: QueueManager, args) -> int:
    """Save a template to the bank."""
    file_path = manager.save_prompt_to_bank(args.template_name, args.priority)
    print(f"✓ Created template in bank: {file_path}")
    print(f"Edit {file_path} to customize your template")
    return 0


def cmd_bank_list(manager: QueueManager, args) -> int:
    """List templates in the bank."""
    templates = manager.list_bank_templates()
    
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


def cmd_bank_use(manager: QueueManager, args) -> int:
    """Use a template from the bank."""
    success = manager.use_bank_template(args.template_name)
    return 0 if success else 1


def cmd_bank_delete(manager: QueueManager, args) -> int:
    """Delete a template from the bank."""
    success = manager.delete_bank_template(args.template_name)
    return 0 if success else 1


def cmd_prompt_box(args) -> int:
    """Launch the interactive prompt box CLI."""
    try:
        # With setuptools-rust bins, the binary gets installed to the Python environment's bin directory
        # We need to find it using shutil.which or check common locations
        import shutil
        
        binary_name = "prompt-box"
        if sys.platform == "win32":
            binary_name += ".exe"
        
        # Try to find the binary in PATH first
        binary_path = shutil.which(binary_name)
        
        if not binary_path:
            # Fallback: check in the same directory as the Python executable
            python_bin_dir = os.path.dirname(sys.executable)
            potential_path = os.path.join(python_bin_dir, binary_name)
            if os.path.exists(potential_path):
                binary_path = potential_path
        
        if not binary_path or not os.path.exists(binary_path):
            print(f"Error: prompt-box binary not found. Please reinstall the package.")
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
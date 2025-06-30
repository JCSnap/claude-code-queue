#!/usr/bin/env python3
"""
Claude Code Queue - Main CLI entry point.

A tool to queue Claude Code prompts and automatically execute them when token limits reset.
"""

import json
from datetime import datetime
from typing import List, Optional

import typer
from typing_extensions import Annotated

from .queue_manager import QueueManager
from .models import QueuedPrompt, PromptStatus

app = typer.Typer(
    name="python -m claude_code_queue.cli",
    help="Claude Code Queue - Queue prompts and execute when limits reset.",
    rich_markup_mode="markdown",
    epilog="""
Examples:
  \b
  # Start the queue processor
  python -m claude_code_queue.cli start
  \b
  # Add a quick prompt
  python -m claude_code_queue.cli add "Fix the authentication bug" --priority 1
  \b
  # Create a template for detailed prompt
  python -m claude_code_queue.cli template my-feature --priority 2
  \b
  # Check queue status
  python -m claude_code_queue.cli status
  \b
  # Cancel a prompt
  python -m claude_code_queue.cli cancel abc123
  \b
  # Test Claude Code connection
  python -m claude_code_queue.cli test
    """,
)

# Shared state object to hold the QueueManager instance
state = {}

@app.callback()
def main(
    ctx: typer.Context,
    storage_dir: Annotated[str, typer.Option(help="Storage directory for queue data.")] = "~/.claude-queue",
    claude_command: Annotated[str, typer.Option(help="Claude Code CLI command.")] = "claude",
    check_interval: Annotated[int, typer.Option(help="Check interval in seconds.")] = 30,
    timeout: Annotated[int, typer.Option(help="Command timeout in seconds.")] = 3600,
):
    """
    Claude Code Queue - A tool to queue Claude Code prompts and automatically execute them when token limits reset.
    """
    try:
        # Store manager instance in the shared state object
        state["manager"] = QueueManager(
            storage_dir=storage_dir,
            claude_command=claude_command,
            check_interval=check_interval,
            timeout=timeout,
        )
    except Exception as e:
        typer.secho(f"Error initializing: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def start(
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Verbose output.")] = False
):
    """Start the queue processor."""
    manager: QueueManager = state["manager"]
    status_callback = None
    if verbose:
        def callback(q_state):
            stats = q_state.get_stats()
            typer.echo(f"Queue status: {stats['status_counts']}")
        status_callback = callback

    try:
        manager.start(callback=status_callback)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user. Shutting down.")
    except Exception as e:
        typer.secho(f"Error during execution: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def add(
    prompt: Annotated[str, typer.Argument(help="The prompt text")],
    priority: Annotated[int, typer.Option("-p", "--priority", help="Priority (lower = higher priority).")] = 0,
    working_dir: Annotated[str, typer.Option("-d", "--working-dir", help="Working directory.")] = ".",
    context_files: Annotated[Optional[List[str]], typer.Option("-f", "--context-files", help="Context files to include.")] = None,
    max_retries: Annotated[int, typer.Option("-r", "--max-retries", help="Maximum retry attempts.")] = 3,
    estimated_tokens: Annotated[Optional[int], typer.Option("-t", "--estimated-tokens", help="Estimated token usage.")] = None,
):
    """Add a prompt to the queue."""
    manager: QueueManager = state["manager"]
    queued_prompt = QueuedPrompt(
        content=prompt,
        working_directory=working_dir,
        priority=priority,
        context_files=context_files or [],
        max_retries=max_retries,
        estimated_tokens=estimated_tokens,
    )

    if manager.add_prompt(queued_prompt):
        typer.secho(f"Successfully added prompt {queued_prompt.id}", fg=typer.colors.GREEN)
    else:
        typer.secho("Failed to add prompt.", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def template(
    filename: Annotated[str, typer.Argument(help="Template filename (without .md extension).")],
    priority: Annotated[int, typer.Option("-p", "--priority", help="Default priority.")] = 0,
):
    """Create a prompt template file."""
    manager: QueueManager = state["manager"]
    file_path = manager.create_prompt_template(filename, priority)
    typer.echo(f"Created template: {file_path}")
    typer.echo("Edit the file and it will be automatically picked up by the queue processor.")


@app.command()
def status(
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
    detailed: Annotated[bool, typer.Option("-d", "--detailed", help="Show detailed prompt info.")] = False,
):
    """Show queue status."""
    manager: QueueManager = state["manager"]
    q_state = manager.get_status()
    stats = q_state.get_stats()

    if json_output:
        typer.echo(json.dumps(stats, indent=2))
        return

    typer.secho("Claude Code Queue Status", bold=True)
    typer.echo("=" * 40)
    typer.echo(f"Total prompts: {stats['total_prompts']}")
    typer.echo(f"Total processed: {stats['total_processed']}")
    typer.echo(f"Failed count: {stats['failed_count']}")
    typer.echo(f"Rate limited count: {stats['rate_limited_count']}")

    if stats["last_processed"]:
        last_processed = datetime.fromisoformat(stats["last_processed"])
        typer.echo(f"Last processed: {last_processed.strftime('%Y-%m-%d %H:%M:%S')}")

    typer.secho("\nStatus breakdown:", bold=True)
    for status_val, count in stats["status_counts"].items():
        if count > 0:
            typer.echo(f"  {status_val}: {count}")

    if stats["current_rate_limit"]["is_rate_limited"]:
        reset_time = stats["current_rate_limit"]["reset_time"]
        if reset_time:
            reset_dt = datetime.fromisoformat(reset_time)
            typer.secho(f"\nRate limited until: {reset_dt.strftime('%Y-%m-%d %H:%M:%S')}", fg=typer.colors.YELLOW)

    if detailed and q_state.prompts:
        typer.secho("\nPrompts:", bold=True)
        typer.echo("-" * 40)
        status_icons = {
            PromptStatus.QUEUED: "⏳", PromptStatus.EXECUTING: "▶️", PromptStatus.COMPLETED: "✅",
            PromptStatus.FAILED: "❌", PromptStatus.CANCELLED: "🚫", PromptStatus.RATE_LIMITED: "⚠️",
        }
        for prompt in sorted(q_state.prompts, key=lambda p: p.priority):
            icon = status_icons.get(prompt.status, "❓")
            typer.echo(f"{icon} {prompt.id} (P{prompt.priority}) - {prompt.status.value}")
            typer.echo(f"   {prompt.content[:80]}{'...' if len(prompt.content) > 80 else ''}")
            if prompt.retry_count > 0:
                typer.echo(f"   Retries: {prompt.retry_count}/{prompt.max_retries}")


@app.command()
def cancel(
    prompt_id: Annotated[str, typer.Argument(help="Prompt ID to cancel.")]
):
    """Cancel a prompt."""
    manager: QueueManager = state["manager"]
    if manager.remove_prompt(prompt_id):
        typer.secho(f"Successfully cancelled prompt {prompt_id}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Could not find or cancel prompt {prompt_id}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command(name="list")
def list_prompts(
    status: Annotated[Optional[PromptStatus], typer.Option(
        help="Filter by status.",
        case_sensitive=False
    )] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
):
    """List prompts."""
    manager: QueueManager = state["manager"]
    q_state = manager.get_status()
    prompts = q_state.prompts

    if status:
        prompts = [p for p in prompts if p.status == status]

    if json_output:
        prompt_data = [json.loads(p.model_dump_json()) for p in prompts]
        typer.echo(json.dumps(prompt_data, indent=2))
        return

    if not prompts:
        typer.echo("No prompts found.")
        return

    typer.echo(f"Found {len(prompts)} prompts:")
    typer.echo("-" * 80)
    status_icons = {
        PromptStatus.QUEUED: "⏳", PromptStatus.EXECUTING: "▶️", PromptStatus.COMPLETED: "✅",
        PromptStatus.FAILED: "❌", PromptStatus.CANCELLED: "🚫", PromptStatus.RATE_LIMITED: "⚠️",
    }
    for prompt in sorted(prompts, key=lambda p: (p.priority, p.created_at)):
        icon = status_icons.get(prompt.status, "❓")
        typer.echo(f"{icon} {prompt.id} | P{prompt.priority} | {prompt.status.value}")
        typer.echo(f"   {prompt.content[:70]}{'...' if len(prompt.content) > 70 else ''}")
        typer.echo(f"   Created: {prompt.created_at.strftime('%Y-%m-%d %H:%M:%S')}")


@app.command()
def test():
    """Test Claude Code connection."""
    manager: QueueManager = state["manager"]
    is_working, message = manager.claude_interface.test_connection()
    color = typer.colors.GREEN if is_working else typer.colors.RED
    typer.secho(message, fg=color)
    if not is_working:
        raise typer.Exit(1)

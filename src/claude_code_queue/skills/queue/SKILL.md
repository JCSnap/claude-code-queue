---
name: queue
description: >
  Use when the user wants to queue a task for Claude Code, add prompts to
  claude-queue, create prompt template files with YAML frontmatter, check
  queue status, manage the prompt bank, process batch jobs, or when a task
  is complex enough that it may hit API rate limits and should run
  automatically with retry. Triggers on: "queue this", "add to queue",
  "run later", "rate limit", "prompt bank", "save as template",
  "batch of tasks", "claude-queue".
allowed-tools: [Bash, Read, Glob]
argument-hint: "[task description or subcommand]"
disable-model-invocation: false
---

# Claude Queue Skill

`claude-queue` queues prompts for Claude Code with automatic rate-limit
retry. Use it proactively when tasks are complex, rate-limit-prone, or
when the user asks to queue something.

## When to Proactively Suggest Queuing

Suggest using claude-queue when you notice:
- The task involves many sequential tool calls that may hit the usage limit
- The user says "run this later" or "I'll come back to this"
- The user describes a batch of similar tasks (multiple files, features, etc.)
- A rate limit was already hit in this session

Example: "This refactor spans many files and may hit API limits midway.
Want me to queue it with `claude-queue` so it retries automatically?"

## Core Commands

```bash
# Quick add (short prompts, no context files needed)
claude-queue add "Fix the auth bug" --priority 1 --working-dir /path/to/project

# Create a detailed template file (complex prompts with context)
claude-queue template task-name --priority 2

# Check queue
claude-queue status --detailed
claude-queue list --status queued

# Start the daemon (runs prompts, auto-retries on rate limit)
claude-queue start

# Cancel a queued prompt
claude-queue cancel <prompt-id>
```

## Prompt Template Format

Template files live at `~/.claude-queue/queue/task-name.md`:

```markdown
---
priority: 0
working_directory: /absolute/path/to/project
context_files:
  - src/relevant-file.py
  - tests/test_relevant.py
max_retries: 3
estimated_tokens: null
---

# Task Title

Clear, self-contained description of what Claude Code should do.
Write as if there is no prior conversation context.

## Context

Background, constraints, or requirements.

## Expected Output

What should be delivered when done.
```

### Frontmatter Fields

| Field | Notes |
|---|---|
| `priority` | **0 = highest**. Lower number executes first. |
| `working_directory` | Absolute path. Use the actual project path from context. |
| `context_files` | Paths relative to `working_directory`. Only include files that exist. |
| `max_retries` | Total attempts: `3` = 3 total, `-1` = unlimited, `1` = no retry. Rate-limit retries and failures share this counter. |
| `estimated_tokens` | Optional hint; set `null` if unknown. |

### Priority Guidelines

- `0` — critical / blocks other work
- `1` — high priority (default for explicit user requests)
- `2` — normal work
- `5+` — background / batch jobs

## Deciding: `add` vs. Template File

**Use `claude-queue add`** when:
- Prompt is short (~200 chars or less)
- No context files needed
- Single, clear action

**Use a template file** when:
- Multi-step task needing structured instructions
- Context files should be attached
- The user will want to review or edit before running

For template files: construct the content and either run
`claude-queue template task-name` and show the user what to paste in, or
write the YAML+markdown content directly and tell the user to save it to
`~/.claude-queue/queue/task-name.md`.

## Template Bank (Reusable Templates)

```bash
claude-queue bank save daily-docs      # Create template in bank
claude-queue bank list                  # List saved templates
claude-queue bank use daily-docs        # Copy to active queue with new ID
claude-queue bank delete daily-docs     # Delete from bank
```

Bank templates live at `~/.claude-queue/bank/` and use the same YAML
frontmatter format. `bank use` copies them to the active queue —
templates themselves never execute directly.

## Batch Processing

For running the same task across many inputs:

```bash
claude-queue batch generate my-template --data entries.csv
claude-queue batch generate my-template --data entries.csv --dry-run
claude-queue batch variables my-template    # list {{placeholders}} in template
claude-queue batch validate my-template --data entries.csv
```

Batch templates use `{{variable}}` placeholders that match CSV column names:

```markdown
---
priority: 2
working_directory: /path/to/project
context_files: []
max_retries: 3
estimated_tokens: null
---

Refactor `{{filename}}` located at `{{filepath}}`:
- Update all calls from old_api() to new_api()
- Preserve existing tests
```

## Queue Directory Layout

```
~/.claude-queue/
├── queue/           # Active prompts (daemon reads these)
├── completed/       # Successfully finished
├── failed/          # Exhausted max_retries
├── bank/            # Saved reusable templates
└── queue-state.json
```

## Key Behavior Notes

**`--dangerously-skip-permissions`**: Passed to `claude` by default so the
daemon runs unattended. To disable interactive permission prompts:
`claude-queue start --no-skip-permissions`.

**At-least-once semantics**: If the daemon crashes mid-execution, the task
reruns on restart. Prompt users to design queued tasks to be idempotent
(safe to run twice) — especially for actions like "send email", "open PR",
"create payment", etc.

**Always suggest** running `claude-queue status --detailed` before starting
the daemon so the user can review what will execute.

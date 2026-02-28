# Claude Code Queue

A tool to queue Claude Code prompts and automatically execute them when token limits reset, preventing manual waiting during 5-hour limit windows.

## Features

-   **Markdown-based Queue**: Each prompt is a `.md` file with YAML frontmatter
-   **Automatic Rate Limit Handling**: Detects rate limits and waits for reset windows
-   **Priority System**: Execute high-priority prompts first
-   **Retry Logic**: Automatically retry failed prompts
-   **Persistent Storage**: Queue survives system restarts
-   **Prompt Bank**: Save and reuse templates for recurring tasks
-   **Interactive Prompt Box**: Browse and select files interactively with fuzzy search
-   **CLI Interface**: Simple command-line interface

## Installation

```bash
pip install claude-code-queue
```

**Note**: The interactive prompt box is built with Rust. If installing from source (not from PyPI), you'll need the [Rust toolchain](https://rustup.rs/) installed.

Or, for local development:

```bash
cd claude-code-queue
pip install -e .
```

## Quick Start

After installation, use the `claude-queue` command:

1. **Test Claude Code connection:**

    ```bash
    claude-queue test
    ```

2. **Add a quick prompt:**

    ```bash
    claude-queue add "Fix the authentication bug" --priority 1
    ```

3. **Create a detailed prompt template:**

    ```bash
    claude-queue template my-feature --priority 2
    # Edit ~/.claude-queue/queue/my-feature.md with your prompt
    ```

4. **Launch the interactive prompt box:**
    ```bash
    claude-queue prompt-box
    ```

5. **Start the queue processor:**
    ```bash
    claude-queue start
    ```

## Usage

### Adding Prompts

**Quick prompt:**

```bash
claude-queue add "Implement user authentication" --priority 1 --working-dir /path/to/project
```

**Template for detailed prompt:**

```bash
claude-queue template auth-feature
```

This creates `~/.claude-queue/queue/auth-feature.md`:

```markdown
---
priority: 0
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
```

### Managing the Queue

**Check status:**

```bash
claude-queue status --detailed
```

**List prompts:**

```bash
claude-queue list --status queued
```

**Cancel a prompt:**

```bash
claude-queue cancel abc123
```

### Running the Queue

**Start processing:**

```bash
claude-queue start
```

**Start with verbose output:**

```bash
claude-queue start --verbose
```

## Prompt Bank (Template Management)

The Prompt Bank allows you to save and reuse templates for recurring tasks like daily documentation updates, weekly reports, or standard maintenance tasks.

### Saving Templates to Bank

**Create a new template in the bank:**

```bash
claude-queue bank save update-docs --priority 1
```

This creates `~/.claude-queue/bank/update-docs.md` which you can edit:

```markdown
---
priority: 1
working_directory: /path/to/project
context_files:
  - README.md
  - docs/
max_retries: 3
estimated_tokens: 1500
---

# Update Project Documentation

Please review and update the project documentation:

## Tasks
1. Update README.md with latest features
2. Check code examples are current
3. Update API documentation
4. Fix any broken links

## Context
This is the daily documentation review task.
```

### Managing Templates

**List available templates:**

```bash
claude-queue bank list
```

**Use a template (adds to queue):**

```bash
claude-queue bank use update-docs
```

**Delete a template:**

```bash
claude-queue bank delete update-docs
```

### Typical Workflow for Recurring Tasks

1. **One-time setup:**
   ```bash
   # Create and customize your template
   claude-queue bank save daily-standup --priority 1
   # Edit ~/.claude-queue/bank/daily-standup.md with your specific requirements
   ```

2. **Daily usage:**
   ```bash
   # Simply add to queue whenever needed
   claude-queue bank use daily-standup
   claude-queue start
   ```

This eliminates the need to recreate the same prompt structure every time!

## Interactive Prompt Box

The prompt box provides an interactive terminal UI for browsing and selecting files with fuzzy search capabilities.

### Launching the Prompt Box

```bash
claude-queue prompt-box
```

### Features

- **Fuzzy File Search**: Type to filter files by name or path
- **Real-time Preview**: See file contents as you navigate
- **Keyboard Navigation**: Use arrow keys to browse files
- **File Selection**: Select files to include in prompts
- **Directory Traversal**: Browse through project directories
- **Copy to Clipboard**: Copy file paths or contents

### Keyboard Shortcuts

**Input Mode:**
- `↑/↓`: Navigate history
- `←/→`: Move cursor left/right
- `Home/End`: Move cursor to beginning/end
- `Enter`: Submit input
- `Tab`: Trigger file picker for @ mentions
- `Ctrl+C`: Copy to clipboard
- `Ctrl+Q`: Quit application

**Picker Mode (file selection):**
- `↑/↓`: Navigate through files
- `Enter`: Select current file
- `Esc`: Exit picker mode
- Type to search files with fuzzy matching

The prompt box is built with Rust for fast file indexing and responsive UI, making it easy to explore large codebases and select relevant files for your Claude Code prompts.

## How It Works

1. **Queue Processing**: Runs prompts in priority order (lower number = higher priority)
2. **Rate Limit Detection**: Monitors Claude Code output for rate limit messages
3. **Automatic Waiting**: When rate limited, parses the actual reset time from Claude's output when available; falls back to estimating the next 5-hour window boundary otherwise
4. **Retry Logic**: Failed prompts are retried up to `max_retries` total attempts. Interrupted prompts (from crashes or ungraceful shutdowns) are automatically re-queued on the next startup — **at-least-once semantics apply**: a task that finished but whose result was not saved before a crash will run again. Design tasks to be idempotent where possible.
5. **File Organization**:
    - `~/.claude-queue/queue/` - Pending prompts
    - `~/.claude-queue/completed/` - Successful executions
    - `~/.claude-queue/failed/` - Failed prompts
    - `~/.claude-queue/bank/` - Saved template library
    - `~/.claude-queue/queue-state.json` - Queue metadata
6. **Execution Logs**: After each execution attempt, a log is appended to the prompt's `.md` file for human inspection. The log is automatically stripped before the prompt is sent to Claude, so Claude always receives only the original prompt text regardless of how many times the task has been retried.

## Configuration

### Command Line Options

```bash
claude-queue --help
```

Key options:

-   `--storage-dir`: Queue storage location (default: `~/.claude-queue`)
-   `--claude-command`: Claude CLI command (default: `claude`)
-   `--check-interval`: Check interval in seconds (default: 30)
-   `--timeout`: Command timeout in seconds (default: 3600)

### Prompt Configuration

Each prompt supports these YAML frontmatter options:

```yaml
---
priority: 1 # Execution priority (0 = highest)
working_directory: /path/to/project # Where to run the prompt
context_files: # Files to include as context
    - src/main.py
    - README.md
max_retries: 3 # Maximum total execution attempts (1 = no retries, -1 = unlimited)
estimated_tokens: 1000 # Estimated token usage (optional)
---
```

**`max_retries` semantics:** this field controls the total number of execution attempts, not the number of retries after the first failure. `max_retries: 3` means 3 total attempts (initial + 2 retries); `max_retries: 1` means a single attempt with no retries; `max_retries: -1` means unlimited retries. Rate-limited executions and failure retries share the same counter.

## Examples

### Basic Usage

```bash
# Add a simple prompt
claude-queue add "Run tests and fix any failures" --priority 1

# Create template for complex prompt
claude-queue template database-migration --priority 2

# Launch interactive file browser
claude-queue prompt-box

# Save a reusable template
claude-queue bank save update-docs --priority 1

# Use a saved template
claude-queue bank use update-docs

# Start processing
claude-queue start
```

### Complex Prompt Template

```markdown
---
priority: 1
working_directory: /Users/me/my-project
context_files:
    - src/auth.py
    - tests/test_auth.py
    - docs/auth-requirements.md
max_retries: 2
estimated_tokens: 2000
---

# Fix Authentication Bug

There's a bug in the user authentication system where users can't log in with special characters in their passwords.

## Context

-   The issue affects passwords containing @, #, $ symbols
-   Error occurs in the password validation function
-   Tests are failing in test_auth.py

## Requirements

1. Fix the password validation to handle special characters
2. Update tests to cover edge cases
3. Ensure backward compatibility

## Expected Output

-   Fixed authentication code
-   Updated test cases
-   Documentation update if needed
```

## Rate Limit Handling

The system automatically detects Claude Code rate limits by monitoring:

-   "usage limit reached" messages
-   Claude's reset time information
-   Standard rate limit error patterns

When rate limited:

1. Prompt status changes to `rate_limited`
2. The queue determines the reset time using a two-tier strategy:
    - **Parsed reset time**: extracts the actual reset time from Claude's output when available
    - **Estimated reset time**: falls back to estimating the next 5-hour window boundary (00:00–05:00, 05:00–10:00, 10:00–15:00, 15:00–20:00, 20:00–01:00) based on the current time
3. Once the reset time is reached, the prompt is re-queued and execution resumes

## Troubleshooting

**Queue not processing:**

```bash
# Check Claude Code connection
claude-queue test

# Check queue status
claude-queue status --detailed
```

**Prompts stuck in executing state:**

Interrupted prompts are **automatically re-queued** on the next startup — no manual intervention is needed. Simply restart the queue:

```bash
claude-queue start
```

> **Warning — at-least-once execution:** If the daemon was killed after Claude finished but before the result was saved to disk, the task will run again on restart. Design tasks to be idempotent (safe to run more than once) where possible.

**Rate limit not detected:**

-   Check if Claude Code output format changed
-   File an issue with the error message you received

## Directory Structure

```
~/.claude-queue/
├── queue/               # Pending prompts
│   ├── 001-fix-bug.md
│   └── 002-feature.executing.md
├── completed/           # Successful executions
│   └── 001-fix-bug-completed.md
├── failed/              # Failed prompts
│   └── 003-failed-task.md
├── bank/                # Saved template library
│   ├── update-docs.md
│   ├── daily-standup.md
│   └── weekly-report.md
└── queue-state.json     # Queue metadata
```

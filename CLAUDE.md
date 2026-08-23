# CLAUDE.md — Claude Code Queue

## Project Overview

Queue Claude Code prompts and execute them automatically when token limits reset.
Markdown-based persistent queue with YAML frontmatter, automatic rate-limit
detection, priority scheduling, and retry logic.

**Version**: 0.5.0
**Python**: >=3.8
**Only runtime dependency**: PyYAML >= 6.0
**Entry point**: `claude-queue` (console script via `claude_code_queue.cli:main`)

## Architecture

Six core modules in `src/claude_code_queue/`:

| Module | Responsibility |
|---|---|
| `models.py` | Data structures: `PromptStatus` enum, `QueuedPrompt`, `RateLimitInfo`, `QueueState`, `ExecutionResult` |
| `storage.py` | File I/O: `MarkdownPromptParser` (YAML frontmatter ↔ dataclass), `QueueStorage` (directory management, cross-dir moves) |
| `queue_manager.py` | Execution loop: `QueueManager` (iteration, retry, rate-limit checking, signal handling, graceful shutdown) |
| `claude_interface.py` | Subprocess management: `ClaudeCodeInterface` (Popen, rate-limit detection, SIGTERM→SIGKILL escalation, atexit cleanup) |
| `batch.py` | Template rendering: `{{variable}}` replacement from CSV/TSV, batch job generation |
| `cli.py` | Argparse CLI: dispatches subcommands; storage-only commands avoid `claude` binary dependency (E3 pattern) |

### Data Flow

```
CLI (add/template/bank)
  → QueueStorage._save_single_prompt()  → writes .md to queue/

CLI (start)
  → QueueManager.start()
    → _process_queue_iteration() loop
      → storage.load_queue_state()       (re-reads all .md files every tick)
      → _check_rate_limited_prompts()    (RATE_LIMITED → QUEUED or FAILED)
      → state.get_next_prompt()          (priority-ordered, cooldown-aware)
      → _execute_prompt()
        → claude_interface.execute_prompt()  (subprocess with --print)
      → _process_execution_result()      (SUCCESS/RATE_LIMITED/FAILED routing)
      → storage.save_queue_state()       (writes files + queue-state.json)
```

### File-Based Queue

```
~/.claude-queue/
├── queue/              Active prompts (.md, .executing.md, .rate-limited.md)
├── completed/          Finished prompts
├── failed/             Failed and cancelled prompts
├── bank/               Reusable templates
└── queue-state.json    Persistent counters (total_processed, failed_count, etc.)
```

**Filename convention**: `{id}-{sanitized-title}.md` with status suffixes
(`.executing.md`, `.rate-limited.md`). The `{id}` is 8 hex chars from uuid4.

**File permissions**: dirs 0o700, files 0o600 (SC3 hardening).

## Key Design Decisions

### At-Least-Once Semantics
`.executing.md` files left on disk after a crash are re-queued on next startup.
A task that completed but whose result wasn't saved before a crash will run again.
Tasks should be idempotent where possible.

### Rate-Limit Detection
1. **Stderr** (primary): first 2048 chars scanned for patterns
2. **Stdout tail** (fallback, Fix 2): only when returncode != 0 and stderr has no match; uses stdout-safe patterns only (no broad patterns to avoid false positives from subprocess output)
3. **Reset time**: parsed from `usage limit reached|<unix_ts>`, ISO timestamps, or estimated from 5-hour UTC boundaries (00, 05, 10, 15, 20)
4. **Cap**: max 24 hours into the future (SC4 — prevents queue stall)

### Resuming Interrupted Work
A queued job that is interrupted — a usage limit, a crash, a timeout — continues
its conversation on the next attempt instead of starting over.

- The manager persists a queue-owned `session_id` before launch. A recovered
  attempt resumes it when its exact transcript exists; otherwise it starts that
  same reserved UUID with `--session-id`. A resumed attempt sends the resume
  message rather than the original instruction. Re-sending the
  instruction would invite redoing finished work; context files are skipped for
  the same reason (they are already in the conversation).
- `--resume` reuses the same session, so one prompt yields one conversation log
  however many times it is interrupted.
- Both `--session-id` and `--resume` are feature-detected once at startup from
  `--help` (`_detect_supported_flags()`). A CLI predating either flag rejects it
  outright. A continuation fails without launching when `--resume` is unavailable;
  silently starting it as a new task can repeat destructive work.
- `claude-queue sessions` lists conversations newest first with the id
  `resume-session` needs. `sessions.py` reads only each transcript's opening
  records — the generated `aiTitle` sits around line 11, so a line and byte cap
  with an early exit keeps a listing from reading hundreds of megabytes. Project
  filtering matches the `cwd` recorded inside the log rather than decoding Claude
  Code's directory name, which rewrites `.` and `_` as well as `/`.
- `claude-queue resume-session [SESSION_ID]` queues a continuation of an existing
  conversation, defaulting to `$CLAUDE_CODE_SESSION_ID` so it can be run from
  inside the session that hit the limit. The continuation runs in the directory
  the session itself recorded, not the caller's — resuming a conversation
  somewhere else would point it at the wrong files. `-d` overrides that. The continuation runs non-interactively
  via `--print`; the session stays reopenable with `claude --resume <id>`.

### Resume Message Resolution
`config.resolve_resume_message()`, most specific first:

1. the prompt's `resume_message` frontmatter
2. `.claude-queue.yaml` in the prompt's working directory (project)
3. `config.yaml` in the storage directory (queue-wide, so per-profile)
4. `config.DEFAULT_RESUME_MESSAGE`

Blank and non-string values count as unset and fall through, so clearing a field
never resumes with an empty prompt. A malformed config warns on stderr and falls
back — a daemon that refuses to start over a stray tab in YAML is worse than one
running on defaults.

### Session Artifact Cleanup
Each run leaves a todo stub, a debug transcript, and telemetry events under the
Claude Code config directory. `_do_cleanup_session_artifacts()` removes them when
a prompt reaches a terminal state (COMPLETED or FAILED).

- **The conversation log is kept.** While the prompt is retryable it is the state
  `--resume` continues from; afterwards it is the record of what the run did.
  Because retries resume, logs no longer accumulate per attempt.
- **Correlation is exact, never heuristic.** Every path is built from the session
  UUID the queue generated, so no size or mtime guessing is involved and no other
  session can match.
- **Imported sessions are not cleaned.** Their UUID can name artifacts created
  before the queue owned the continuation.
- **Config directory honours `$CLAUDE_CONFIG_DIR`** (`paths.claude_config_dir()`).
  Hardcoding `~/.claude` makes cleanup a silent no-op for anyone using a custom
  config directory, and installs skills into the wrong profile.
- Depends on undocumented Claude Code internals (`todos/`, `debug/`,
  `telemetry/`). If the layout changes, cleanup stops finding files — safe, since
  nothing outside these session-scoped names is ever touched. Failures are logged
  and swallowed so `save_queue_state()` always runs.

### Per-Prompt Profiles and Per-Account Limits
Each Claude Code config directory carries its own credentials, so the profile a
prompt records decides which account it bills to.

- `QueuedPrompt.claude_config_dir` is persisted in frontmatter and set at queue
  time by `cli._resolve_profile()` — the active `$CLAUDE_CONFIG_DIR` unless
  `--profile` overrides it. `execute_prompt()` puts it in the subprocess
  environment; without it a prompt silently spends whichever account the
  processor started under.
- `QueuedPrompt.profile_key()` resolves an unset value to the active config
  directory, so a prompt queued before profiles existed groups with the account
  it actually bills to rather than looking like a separate one.
- **Usage limits are per account**, so `QueueState.get_next_prompt()` blocks only
  the profiles whose reset window has not arrived. Work billed to another account
  keeps running — the reason for queueing across profiles at all.
- Artifact cleanup targets the prompt's profile, which is not necessarily the
  processor's.

### Session Usage Statistics
After each execution, the queue prints the duration and reads token usage from
the exact session JSONL file. `ExecutionResult.session_id` selects the file in
the prompt's canonical `claude_config_dir` profile. The parser counts each
assistant message ID once because Claude can store several events for one API
response. It adds non-cached, cache-write, and cache-read tokens for the displayed
input total. A durable cumulative cursor makes every result report only usage not
reported by an earlier attempt. Usage from a killed attempt appears with the next
completed result. Imported sessions establish their historical usage as the
pre-launch baseline. The prompt log keeps the breakdown across all result paths.

### Retry Logic
- `max_retries` = total attempts (3 = initial + 2 retries; -1 = unlimited)
- Rate-limit hits and generic failures share the same `retry_count`
- Generic failures get a 60s cooldown via `retry_not_before` (prevents spin loops)
- Non-retryable errors (e.g., nested Claude session) → immediate FAILED, no retry budget consumed
- Terminal status blocklist in `can_retry()`: COMPLETED and CANCELLED never retry

### Single-Writer Lock
`QueueManager.start()` takes an advisory exclusive lock on the storage directory
(`.queue.lock`, `locking.QueueLock`) and refuses to start if another processor
holds it.

- **Why**: each tick reloads every prompt file, claims the highest-priority one,
  and rewrites `queue-state.json` wholesale. Two processors on one directory
  claim the same prompt — executing the task twice — and clobber each other's
  counters. Queued tasks are advised to be idempotent, never required to be.
- **`flock`/`msvcrt`, not a marker file**: the kernel holds the lock for the life
  of the file descriptor, so it is released even on SIGKILL. There is no stale
  lock to clear. The lock file is deliberately *not* unlinked on release —
  unlinking lets a second processor lock an orphaned inode.
- **Scope**: only `start` locks. Storage-only commands (`add`, `status`, `list`,
  `bank`, `batch`) run freely alongside a live processor.
- **Exit status**: `start()` returns False when it declines to start (locked, or
  the claude CLI is unreachable); `cmd_start` maps that to exit code 1 so a
  supervisor sees a failed start as a failure.
- **Running several profiles**: give each its own `--storage-dir`.

### Signal Handling
- SIGINT/SIGTERM → `kill_current()` sends SIGTERM to process group, daemon thread escalates to SIGKILL after 3s
- Signal handler uses `os.write(2, ...)` (not `print()`) to avoid stream-lock re-entrance
- Graceful shutdown reverts EXECUTING → QUEUED with log entry
- `start_new_session=True` isolates subprocess from terminal SIGINT

### CLI Dispatch (E3)
Commands that only read/write files (`add`, `template`, `status`, `list`, `bank`,
`batch`) use `QueueStorage` directly — no `claude` binary needed. Only `start`
and `test` construct `QueueManager` (which verifies the `claude` binary at init).

## Building & Installing

```bash
pip install -e ".[dev]"          # editable install with test deps
pip install -e ".[eval]"         # includes anthropic for LLM eval tests
```

The optional Rust binary `prompt-box` (in `claude-prompt-box/`) requires the
Rust toolchain; the rest of the package works without it.

## Running Tests

```bash
pytest tests/                     # all tests
pytest tests/ -m "not llm_eval"   # skip API-dependent tests
pytest tests/test_fault_tolerance.py -v   # 85+ resilience scenarios
```

**Framework**: pytest + pytest-mock
**Test convention**: test IDs like MOD-001, STO-001, FT-001 for cross-reference

### Fixtures (conftest.py)
- `storage(tmp_path)` — fresh `QueueStorage` in temp dir
- `interface(mocker)` — `ClaudeCodeInterface` with `_verify_claude_available` patched
- `manager(tmp_path, mocker)` — `QueueManager` with both `_verify_claude_available` and `test_connection` patched
- `sample_prompt()` — `QueuedPrompt` with id `"abc12345"`

### Common Mocking Patterns
- `mocker.patch.object(ClaudeCodeInterface, "_verify_claude_available")` — prevent Claude binary check
- `mocker.patch.object(ClaudeCodeInterface, "test_connection", return_value=(True, "ok"))` — simulate connection
- `subprocess.Popen` mock: `proc.pid` must be `int` (not MagicMock), set `proc.communicate.return_value = (stdout, stderr)`
- CLI tests patch `sys.argv` and call `main()` directly

### Marker
`@pytest.mark.llm_eval` — tests requiring `ANTHROPIC_API_KEY`

## Commit Message Convention

```
<type>: <description>
```

Types: `fix:`, `feat:`, `test:`, `chore:`, `docs:`

Fixes reference internal IDs where applicable (e.g., "Fix A", "Fix 3", "SC4", "FT-007").

## Code Conventions

- **Python 3.8+**: dataclasses, f-strings, type hints throughout
- **Type marker**: `py.typed` (PEP 561) for downstream mypy
- **CLI**: argparse (no Click/Typer)
- **No database**: all state is markdown files + one JSON counter file
- **Datetime handling**: always naive (tzinfo stripped on read via `_parse_optional_datetime`); Z-suffix handled for Python < 3.11 compat
- **Filename sanitization**: strips `<>:"/\|?*#'\`` and path traversal components
- **Cross-directory moves**: write destination before deleting source (FT-007)
- **YAML errors**: warn on stderr, fall back to defaults (never crash on bad frontmatter)
- **`retry_not_before`**: cleared via `clear_retry_backoff()` on every status transition except the generic-failure retry path
- Use `python3` (not `python`) when invoking Python in this environment

## YAML Frontmatter Schema

```yaml
---
priority: 0              # Lower = higher priority
working_directory: .     # Execution CWD (resolved relative)
context_files: []        # Files passed as @-references
max_retries: 3           # Total attempts (1=no retry, -1=unlimited)
estimated_tokens: null   # Optional hint
model: null              # Optional Claude model ID
# Internal fields (managed by the queue, not user-edited):
status: queued
retry_count: 0
created_at: <ISO datetime>
last_executed: null
rate_limited_at: null
reset_time: null
retry_not_before: null
session_id: null         # persisted before launch; correlates retries and cleanup
resume_existing_session: false # true only for resume-session jobs
resume_message: null     # overrides the configured resume message
claude_config_dir: null  # canonical absolute profile path
usage_high_water: null   # cumulative usage already reported by this prompt
---
```

## CLI Commands Reference

| Command | Purpose | Needs `claude` binary? |
|---|---|---|
| `start [--verbose] [--no-skip-permissions]` | Run queue loop | Yes |
| `add <prompt> [-p priority] [-m model]` | Quick-add prompt | No |
| `sessions [DIR] [--all] [--search t]` | List session ids and titles | No |
| `resume-session [id] [-m msg]` | Queue a continuation of an existing session | No |
| `template <name> [-p priority]` | Create template .md | No |
| `status [--json] [--detailed]` | Queue stats | No |
| `list [--status <s>] [--json]` | List prompts | No |
| `cancel <id>` | Cancel prompt | No |
| `test` | Verify claude CLI | Yes |
| `bank save/list/use/delete` | Template bank ops | No |
| `batch generate/validate/variables` | Batch job generation | No |
| `cleanup [--dry-run]` | Remove rate-limit artifacts from the active Claude profile | No |
| `install-skill [skill-name] [--force]` | Install bundled skills in the active Claude profile | No |
| `prompt-box` | Launch Rust TUI | No (needs Rust binary) |

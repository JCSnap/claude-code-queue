"""
Interface for executing prompts via Claude Code CLI.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

from .models import ExecutionResult, RateLimitInfo, QueuedPrompt


# Rate-limit messages are written to stderr (not stdout) from this version onward.
_STDERR_RATE_LIMIT_MIN_VERSION = (2, 1, 50)

# Maximum hours into the future a rate-limit reset time may be set.
# Guards against a malicious or buggy claude binary stalling the queue indefinitely.
_RATE_LIMIT_MAX_RESET_HOURS = 24

# S11b — Only scan this many characters of stderr for rate-limit patterns.
# Genuine rate-limit messages from the Claude CLI are short (≤ ~200 chars) and
# appear at the start of stderr. A depth cap prevents a subprocess tool whose
# stderr happens to contain a rate-limit phrase from triggering false positives.
# Value is in Unicode code points (Python string characters); for the ASCII-only
# CLI output this equals the byte count.
_RATE_LIMIT_SCAN_CHARS = 2048

# Only scan this many characters of stderr for non-retryable patterns.
# The nested-session message is short (≤ ~200 chars) and appears near the top
# of stderr. The cap mirrors the _RATE_LIMIT_SCAN_CHARS guard: it prevents a
# subprocess tool whose verbose debug output happens to contain a matching phrase
# from triggering a false positive. Set larger than _RATE_LIMIT_SCAN_CHARS (2 048)
# because future non-retryable patterns may appear after longer preambles.
_NON_RETRYABLE_SCAN_CHARS = 8192

# Errors that are structurally permanent — retrying will never produce a different
# outcome. When any of these substrings is found in the first _NON_RETRYABLE_SCAN_CHARS
# characters of stderr (case-insensitive), the ExecutionResult is marked
# is_non_retryable=True and _process_execution_result() will skip the retry logic
# and immediately mark the prompt FAILED.
#
# Keep patterns specific enough to avoid false positives.
_NON_RETRYABLE_PATTERNS = [
    "claude code cannot be launched inside another claude code session",
]


class ClaudeCodeInterface:
    """Interface for executing prompts via Claude Code CLI."""

    def __init__(self, claude_command: str = "claude", timeout: int = 3600,
                 skip_permissions: bool = True):
        self.claude_command = claude_command
        self.timeout = timeout
        self.skip_permissions = skip_permissions
        self._verify_claude_available()

    def _verify_claude_available(self) -> None:
        """Verify Claude Code CLI is available."""
        try:
            # SC4 Mitigation 3 — If the command is a bare name (not an absolute path),
            # resolve it once at startup using shutil.which() and store the absolute path.
            # Subsequent calls use the absolute path, preventing PATH hijacking between
            # startup and execution. # SC4
            if not Path(self.claude_command).is_absolute():
                resolved = shutil.which(self.claude_command)
                if resolved:
                    self.claude_command = resolved

            subprocess_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            result = subprocess.run(
                [self.claude_command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=subprocess_env,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Claude Code CLI not available: {result.stderr}")

            # R3 — Warn if the installed claude version predates stderr-only rate-limit
            # detection. Older versions may write rate-limit messages to stdout, which
            # would be missed after the R3 change.
            version_str = result.stdout.strip()  # e.g. "2.1.50 (Claude Code)"
            match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
            if match:
                installed = tuple(int(x) for x in match.groups())
                if installed < _STDERR_RATE_LIMIT_MIN_VERSION:
                    print(
                        f"Warning: claude {version_str!r} predates stderr-only rate-limit "
                        f"detection (threshold "
                        f"{'.'.join(str(x) for x in _STDERR_RATE_LIMIT_MIN_VERSION)}). "
                        "Rate-limit messages written to stdout will be missed.",
                        file=sys.stderr,
                    )

        except FileNotFoundError:
            raise RuntimeError(
                f"Claude Code CLI not found. Make sure '{self.claude_command}' is in PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code CLI verification timed out.")

    def execute_prompt(self, prompt: QueuedPrompt) -> ExecutionResult:
        """Execute a prompt via Claude Code CLI."""
        start_time = time.time()

        try:
            working_dir = Path(prompt.working_directory).resolve()

            # SC2 — Warn if working_directory is outside the user's home directory.
            # Mitigates path traversal via YAML-controlled working_directory field.
            # Validation happens BEFORE mkdir() so no directory is created silently.
            # Path.home() raises RuntimeError on containers or headless systems where
            # $HOME is unset; the second except clause skips the check in that case.
            try:
                home = Path.home()
                working_dir.relative_to(home)
            except ValueError:
                print(
                    f"Warning: working_directory {working_dir} is outside home directory "
                    f"({home}). Proceeding with caution.",
                    file=sys.stderr,
                )
            except RuntimeError:
                pass  # home directory not determinable; skip the check

            if not working_dir.exists():
                working_dir.mkdir(parents=True, exist_ok=True)

            # SC1 — Build command conditionally based on skip_permissions setting.
            cmd = [self.claude_command, "--print"]
            if self.skip_permissions:
                cmd.append("--dangerously-skip-permissions")

            full_prompt = prompt.content

            if prompt.context_files:
                context_refs = []
                for context_file in prompt.context_files:
                    # E1 — Resolve context paths against working_dir so the
                    # Python-side exists() guard works correctly for relative paths.
                    # Before E1 (os.chdir), relative paths resolved against the
                    # changed CWD; now we must be explicit.
                    context_path = working_dir / context_file
                    if context_path.exists():
                        context_refs.append(f"@{context_file}")

                if context_refs:
                    full_prompt = f"{' '.join(context_refs)} {prompt.content}"

            cmd.append(full_prompt)

            # E1 — Use cwd= instead of os.chdir() to set the subprocess working directory.
            # This is thread-safe: os.chdir() changes the entire process CWD, which breaks
            # concurrent executions and any other thread that relies on getcwd().
            # Fix A — Strip CLAUDECODE so nested claude invocations are not blocked by the
            # anti-nesting guard. The rest of the environment (PATH, HOME, API keys, etc.)
            # is preserved unchanged.
            subprocess_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
                cwd=str(working_dir), env=subprocess_env
            )

            execution_time = time.time() - start_time

            # R3 — Rate-limit messages from the claude CLI appear on stderr, not stdout.
            # Searching stdout causes false positives if Claude's response happens to
            # contain any trigger phrase (e.g., code that handles rate limits).
            # Caveat: if older versions of the `claude` CLI write rate-limit messages to
            # stdout, this change will miss them (see _STDERR_RATE_LIMIT_MIN_VERSION).
            rate_limit_info = self._detect_rate_limit(result.stderr)
            is_non_retryable = self._detect_non_retryable_error(result.stderr)

            success = result.returncode == 0 and not rate_limit_info.is_rate_limited

            return ExecutionResult(
                success=success,
                output=result.stdout,
                error=result.stderr,
                rate_limit_info=rate_limit_info,
                execution_time=execution_time,
                is_non_retryable=is_non_retryable,
            )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution failed: {str(e)}",
                execution_time=execution_time,
            )

    def _detect_rate_limit(self, output: str) -> RateLimitInfo:
        """Detect rate limiting from Claude Code output."""
        # S11b: restrict pattern scan to the first _RATE_LIMIT_SCAN_CHARS characters.
        # Genuine rate-limit messages from the Claude CLI are short and always appear
        # near the start of stderr. This prevents false positives from rate-limit-like
        # phrases buried in long tool or subprocess output.
        # The reset_extractor and limit_message still use the full output string.
        scan_window = output[:_RATE_LIMIT_SCAN_CHARS].lower()

        # S11a: 'limit exceeded' removed — too broad (matches Python tracebacks, shell
        # ulimit errors, compiler output, etc.). Every meaningful Claude CLI rate-limit
        # scenario is covered by the remaining four patterns.
        # ASSUMPTION: the Claude CLI never uses bare "api limit exceeded" without a
        # "rate", "quota", or "usage" qualifier. Re-verify if CLI output format changes.
        rate_limit_patterns = [
            ("usage limit reached", self._extract_reset_time_from_limit_message),
            ("rate limit exceeded", self._estimate_reset_time),
            ("too many requests", self._estimate_reset_time),
            ("quota exceeded", self._estimate_reset_time),
        ]

        for pattern, reset_extractor in rate_limit_patterns:
            if pattern in scan_window:
                reset_time = reset_extractor(output)
                return RateLimitInfo(
                    is_rate_limited=True,
                    reset_time=reset_time,
                    limit_message=output.strip()[:500],  # First 500 chars
                    timestamp=datetime.now(),
                )

        return RateLimitInfo(is_rate_limited=False)

    def _detect_non_retryable_error(self, stderr: str) -> bool:
        """Return True if stderr contains a non-retryable error pattern.

        Only the first _NON_RETRYABLE_SCAN_CHARS characters of stderr are scanned.
        The nested-session message appears within the first ~200 chars, so the cap
        never misses it; the cap does prevent long tool output from triggering false
        positives if a future pattern is less specific.

        Matching is case-insensitive: each pattern is lowercased at match time, so
        entries in _NON_RETRYABLE_PATTERNS may be any case.
        """
        scan_window = stderr[:_NON_RETRYABLE_SCAN_CHARS].lower()
        return any(pattern.lower() in scan_window for pattern in _NON_RETRYABLE_PATTERNS)

    def _extract_reset_time_from_limit_message(self, output: str) -> Optional[datetime]:
        """Extract reset time from Claude's limit message."""
        try:
            pattern1 = r"usage limit reached\|(\d+)"
            match1 = re.search(pattern1, output, re.IGNORECASE)
            if match1:
                timestamp = int(match1.group(1))
                return ClaudeCodeInterface._cap_reset_time(
                    datetime.fromtimestamp(timestamp)
                )

            pattern2 = (
                r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
            )
            matches = re.findall(pattern2, output)
            if matches:
                latest_time = None
                for match in matches:
                    try:
                        if match.endswith("Z"):
                            ts = datetime.fromisoformat(match.replace("Z", "+00:00"))
                        else:
                            ts = datetime.fromisoformat(match)

                        if latest_time is None or ts > latest_time:
                            latest_time = ts
                    except ValueError:
                        continue

                if latest_time:
                    return ClaudeCodeInterface._cap_reset_time(
                        latest_time + timedelta(hours=5)
                    )

        except Exception as e:
            print(f"Error parsing reset time: {e}")

        return self._estimate_reset_time(output)

    def _estimate_reset_time(self, output: str) -> datetime:
        """Estimate reset time based on Claude's 5-hour windows."""
        now = datetime.now()

        hour = now.hour
        if hour < 5:
            next_reset = now.replace(hour=5, minute=0, second=0, microsecond=0)
        elif hour < 10:
            next_reset = now.replace(hour=10, minute=0, second=0, microsecond=0)
        elif hour < 15:
            next_reset = now.replace(hour=15, minute=0, second=0, microsecond=0)
        elif hour < 20:
            next_reset = now.replace(hour=20, minute=0, second=0, microsecond=0)
        else:  # hour >= 20 → next 5-hour boundary is 01:00 next day
            next_reset = (now + timedelta(days=1)).replace(
                hour=1, minute=0, second=0, microsecond=0
            )

        if next_reset <= now:
            next_reset += timedelta(hours=5)

        # SC4 — Cap to prevent a malicious/buggy claude binary from stalling the queue.
        max_reset = datetime.now() + timedelta(hours=_RATE_LIMIT_MAX_RESET_HOURS)
        return min(next_reset, max_reset)

    @staticmethod
    def _cap_reset_time(dt: datetime) -> datetime:
        """Cap reset time to at most _RATE_LIMIT_MAX_RESET_HOURS from now.

        Also strips timezone info so the returned datetime is always naive,
        consistent with the rest of the codebase. Without stripping, comparing a
        timezone-aware parsed ISO timestamp (e.g. "...+00:00") against naive
        datetime.now() raises TypeError.
        """
        naive_dt = dt.replace(tzinfo=None)
        max_reset = datetime.now() + timedelta(hours=_RATE_LIMIT_MAX_RESET_HOURS)
        return min(naive_dt, max_reset)

    def test_connection(self) -> Tuple[bool, str]:
        """Test if Claude Code is working."""
        try:
            subprocess_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            result = subprocess.run(
                [self.claude_command, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                env=subprocess_env,
            )

            if result.returncode == 0:
                return True, "Claude Code CLI is working"
            else:
                return False, f"Claude Code CLI error: {result.stderr}"

        except FileNotFoundError:
            return False, f"Claude Code CLI not found: {self.claude_command}"
        except subprocess.TimeoutExpired:
            return False, "Claude Code CLI test timed out"
        except Exception as e:
            return False, f"Claude Code CLI test failed: {str(e)}"

    def get_available_commands(self) -> List[str]:
        """Get available Claude Code commands."""
        try:
            subprocess_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            result = subprocess.run(
                [self.claude_command, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                env=subprocess_env,
            )

            if result.returncode == 0:
                lines = result.stdout.split("\n")
                commands = []
                in_commands_section = False

                for line in lines:
                    if "commands:" in line.lower() or "usage:" in line.lower():
                        in_commands_section = True
                        continue

                    if in_commands_section and line.strip():
                        if line.startswith("  "):
                            cmd = line.strip().split()[0]
                            if cmd and not cmd.startswith("-"):
                                commands.append(cmd)

                return commands

        except Exception as e:
            print(f"Error getting available commands: {e}")

        return []

    def execute_simple_prompt(
        self, prompt_text: str, working_dir: str = "."
    ) -> ExecutionResult:
        """Execute a simple prompt without full QueuedPrompt object."""
        simple_prompt = QueuedPrompt(content=prompt_text, working_directory=working_dir)
        return self.execute_prompt(simple_prompt)

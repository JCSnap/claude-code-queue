"""
Data structures for Claude Code Queue system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import uuid


class PromptStatus(Enum):
    """Status of a queued prompt."""

    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"


@dataclass
class QueuedPrompt:
    """Represents a prompt in the queue."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    working_directory: str = "."
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Lower number = higher priority
    context_files: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
    status: PromptStatus = PromptStatus.QUEUED
    execution_log: str = ""
    estimated_tokens: Optional[int] = None
    last_executed: Optional[datetime] = None
    rate_limited_at: Optional[datetime] = None
    reset_time: Optional[datetime] = None

    def add_log(self, message: str) -> None:
        """Add a log entry with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.execution_log += f"[{timestamp}] {message}\n"

    def can_retry(self) -> bool:
        """Check if this prompt can be retried."""
        return self.retry_count < self.max_retries and self.status in [
            PromptStatus.FAILED,
            PromptStatus.RATE_LIMITED,
        ]

    def should_execute_now(self) -> bool:
        """Check if this prompt should be executed now (not rate limited)."""
        if self.status != PromptStatus.RATE_LIMITED:
            return True

        if self.reset_time and datetime.now() >= self.reset_time:
            return True

        return False


@dataclass
class RateLimitInfo:
    """Information about rate limiting from Claude Code response."""

    is_rate_limited: bool = False
    reset_time: Optional[datetime] = None
    limit_message: str = ""
    timestamp: Optional[datetime] = None

    @classmethod
    def from_claude_response(cls, response_text: str) -> "RateLimitInfo":
        """Parse rate limit info from Claude Code response."""
        # Common rate limit indicators in Claude Code responses
        rate_limit_indicators = [
            "usage limit reached",
            "rate limit",
            "too many requests",
            "quota exceeded",
            "limit exceeded",
        ]

        is_limited = any(
            indicator in response_text.lower() for indicator in rate_limit_indicators
        )

        if is_limited:
            return cls(
                is_rate_limited=True,
                limit_message=response_text.strip(),
                timestamp=datetime.now(),
            )

        return cls(is_rate_limited=False)


@dataclass
class QueueState:
    """Overall state of the queue system."""

    prompts: List[QueuedPrompt] = field(default_factory=list)
    last_processed: Optional[datetime] = None
    total_processed: int = 0
    failed_count: int = 0
    rate_limited_count: int = 0
    current_rate_limit: Optional[RateLimitInfo] = None

    def get_next_prompt(self) -> Optional[QueuedPrompt]:
        """Get the next prompt to execute (highest priority, can execute now)."""
        executable_prompts = [
            p
            for p in self.prompts
            if p.status == PromptStatus.QUEUED and p.should_execute_now()
        ]

        if not executable_prompts:
            # Check for rate-limited prompts that can now be retried
            retry_prompts = [
                p
                for p in self.prompts
                if p.status == PromptStatus.RATE_LIMITED
                and p.should_execute_now()
                and p.can_retry()
            ]
            if retry_prompts:
                # Reset status for retry
                prompt = min(retry_prompts, key=lambda p: p.priority)
                prompt.status = PromptStatus.QUEUED
                return prompt

            return None

        # Return highest priority prompt (lowest number)
        return min(executable_prompts, key=lambda p: p.priority)

    def add_prompt(self, prompt: QueuedPrompt) -> None:
        """Add a prompt to the queue."""
        self.prompts.append(prompt)

    def remove_prompt(self, prompt_id: str) -> bool:
        """Remove a prompt from the queue."""
        original_count = len(self.prompts)
        self.prompts = [p for p in self.prompts if p.id != prompt_id]
        return len(self.prompts) < original_count

    def get_prompt(self, prompt_id: str) -> Optional[QueuedPrompt]:
        """Get a prompt by ID."""
        for prompt in self.prompts:
            if prompt.id == prompt_id:
                return prompt
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        status_counts = {}
        for status in PromptStatus:
            if status == PromptStatus.COMPLETED:
                # Use persistent counter for completed prompts
                status_counts[status.value] = self.total_processed
            elif status == PromptStatus.FAILED:
                # Use persistent counter for failed prompts
                status_counts[status.value] = self.failed_count
            else:
                # Count active prompts for other statuses
                status_counts[status.value] = len(
                    [p for p in self.prompts if p.status == status]
                )

        return {
            "total_prompts": len(self.prompts),
            "status_counts": status_counts,
            "total_processed": self.total_processed,
            "failed_count": self.failed_count,
            "rate_limited_count": self.rate_limited_count,
            "last_processed": (
                self.last_processed.isoformat() if self.last_processed else None
            ),
            "current_rate_limit": {
                "is_rate_limited": (
                    self.current_rate_limit.is_rate_limited
                    if self.current_rate_limit
                    else False
                ),
                "reset_time": (
                    self.current_rate_limit.reset_time.isoformat()
                    if self.current_rate_limit and self.current_rate_limit.reset_time
                    else None
                ),
            },
        }


@dataclass
class ExecutionResult:
    """Result of executing a prompt."""

    success: bool
    output: str
    error: str = ""
    rate_limit_info: Optional[RateLimitInfo] = None
    execution_time: float = 0.0

    @property
    def is_rate_limited(self) -> bool:
        """Check if this execution was rate limited."""
        return self.rate_limit_info is not None and self.rate_limit_info.is_rate_limited

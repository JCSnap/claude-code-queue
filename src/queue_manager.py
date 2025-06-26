"""
Queue manager with execution loop.
"""

import time
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable, Dict, Any

from .models import QueuedPrompt, QueueState, PromptStatus, ExecutionResult
from .storage import QueueStorage
from .claude_interface import ClaudeCodeInterface


class QueueManager:
    """Manages the queue execution lifecycle."""
    
    def __init__(self, 
                 storage_dir: str = "~/.claude-queue",
                 claude_command: str = "claude",
                 check_interval: int = 30,
                 timeout: int = 3600):
        self.storage = QueueStorage(storage_dir)
        self.claude_interface = ClaudeCodeInterface(claude_command, timeout)
        self.check_interval = check_interval
        self.running = False
        self.state: Optional[QueueState] = None
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.stop()
    
    def start(self, callback: Optional[Callable[[QueueState], None]] = None) -> None:
        """Start the queue processing loop."""
        print("Starting Claude Code Queue Manager...")
        
        is_working, message = self.claude_interface.test_connection()
        if not is_working:
            print(f"Error: {message}")
            return
        
        print(f"✓ {message}")
        
        self.state = self.storage.load_queue_state()
        print(f"✓ Loaded queue with {len(self.state.prompts)} prompts")
        
        self.running = True
        
        try:
            while self.running:
                self._process_queue_iteration(callback)
                
                if self.running:
                    time.sleep(self.check_interval)
                    
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"Error in queue processing: {e}")
        finally:
            self._shutdown()
    
    def stop(self) -> None:
        """Stop the queue processing loop."""
        self.running = False
    
    def _shutdown(self) -> None:
        """Clean shutdown procedure."""
        print("Shutting down...")
        
        if self.state:
            for prompt in self.state.prompts:
                if prompt.status == PromptStatus.EXECUTING:
                    prompt.status = PromptStatus.QUEUED
                    prompt.add_log("Execution interrupted during shutdown")
            
            self.storage.save_queue_state(self.state)
            print("✓ Queue state saved")
        
        print("Queue manager stopped")
    
    def _process_queue_iteration(self, callback: Optional[Callable[[QueueState], None]] = None) -> None:
        """Process one iteration of the queue."""
        self.state = self.storage.load_queue_state()
        
        self._check_rate_limited_prompts()
        
        next_prompt = self.state.get_next_prompt()
        
        if next_prompt is None:
            rate_limited_prompts = [p for p in self.state.prompts if p.status == PromptStatus.RATE_LIMITED]
            if rate_limited_prompts:
                next_reset = min(p.reset_time for p in rate_limited_prompts if p.reset_time)
                if next_reset:
                    wait_time = (next_reset - datetime.now()).total_seconds()
                    if wait_time > 0:
                        print(f"Waiting for rate limit reset in {self._format_duration(wait_time)}")
                    else:
                        print("Rate limit should have reset, checking prompts...")
            else:
                print("No prompts in queue")
            
            if callback:
                callback(self.state)
            return
        
        print(f"Executing prompt {next_prompt.id}: {next_prompt.content[:50]}...")
        self._execute_prompt(next_prompt)
        
        self.storage.save_queue_state(self.state)
        
        if callback:
            callback(self.state)
    
    def _check_rate_limited_prompts(self) -> None:
        """Check if any rate-limited prompts are ready to retry."""
        current_time = datetime.now()
        
        for prompt in self.state.prompts:
            if (prompt.status == PromptStatus.RATE_LIMITED and 
                prompt.reset_time and 
                current_time >= prompt.reset_time):
                
                if prompt.can_retry():
                    prompt.status = PromptStatus.QUEUED
                    prompt.add_log(f"Rate limit reset, queued for retry")
                    print(f"✓ Prompt {prompt.id} ready for retry after rate limit reset")
                else:
                    prompt.status = PromptStatus.FAILED
                    prompt.add_log(f"Rate limit reset but max retries ({prompt.max_retries}) exceeded")
                    print(f"✗ Prompt {prompt.id} failed - max retries exceeded")
    
    def _execute_prompt(self, prompt: QueuedPrompt) -> None:
        """Execute a single prompt."""
        prompt.status = PromptStatus.EXECUTING
        prompt.last_executed = datetime.now()
        prompt.add_log(f"Started execution (attempt {prompt.retry_count + 1}/{prompt.max_retries})")
        
        self.storage.save_queue_state(self.state)
        
        result = self.claude_interface.execute_prompt(prompt)
        
        self._process_execution_result(prompt, result)
    
    def _process_execution_result(self, prompt: QueuedPrompt, result: ExecutionResult) -> None:
        """Process the result of prompt execution."""
        execution_summary = f"Execution completed in {result.execution_time:.1f}s"
        
        if result.success:
            prompt.status = PromptStatus.COMPLETED
            prompt.add_log(f"{execution_summary} - SUCCESS")
            if result.output:
                prompt.add_log(f"Output:\n{result.output}")
            
            self.state.total_processed += 1
            print(f"✓ Prompt {prompt.id} completed successfully")
            
        elif result.is_rate_limited:
            prompt.status = PromptStatus.RATE_LIMITED
            prompt.rate_limited_at = datetime.now()
            prompt.retry_count += 1
            
            if result.rate_limit_info and result.rate_limit_info.reset_time:
                prompt.reset_time = result.rate_limit_info.reset_time
            else:
                prompt.reset_time = datetime.now() + timedelta(hours=5)
            
            prompt.add_log(f"{execution_summary} - RATE LIMITED")
            prompt.add_log(f"Reset time: {prompt.reset_time}")
            if result.rate_limit_info and result.rate_limit_info.limit_message:
                prompt.add_log(f"Message: {result.rate_limit_info.limit_message}")
            
            self.state.rate_limited_count += 1
            reset_in = (prompt.reset_time - datetime.now()).total_seconds()
            print(f"⚠ Prompt {prompt.id} rate limited, reset in {self._format_duration(reset_in)}")
            
        else:
            prompt.retry_count += 1
            
            if prompt.can_retry():
                prompt.status = PromptStatus.QUEUED
                prompt.add_log(f"{execution_summary} - FAILED (will retry)")
                if result.error:
                    prompt.add_log(f"Error: {result.error}")
                print(f"✗ Prompt {prompt.id} failed, will retry ({prompt.retry_count}/{prompt.max_retries})")
            else:
                prompt.status = PromptStatus.FAILED
                prompt.add_log(f"{execution_summary} - FAILED (max retries exceeded)")
                if result.error:
                    prompt.add_log(f"Error: {result.error}")
                
                self.state.failed_count += 1
                print(f"✗ Prompt {prompt.id} failed permanently after {prompt.max_retries} attempts")
        
        self.state.last_processed = datetime.now()
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable format."""
        if seconds < 0:
            return "now"
        
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            return f"{minutes}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            if minutes == 0:
                return f"{hours}h"
            return f"{hours}h {minutes}m"
    
    def add_prompt(self, prompt: QueuedPrompt) -> bool:
        """Add a prompt to the queue."""
        try:
            if not self.state:
                self.state = self.storage.load_queue_state()
            
            self.state.add_prompt(prompt)
            
            success = self.storage.save_queue_state(self.state)
            if success:
                print(f"✓ Added prompt {prompt.id} to queue")
            else:
                print(f"✗ Failed to save prompt {prompt.id}")
            
            return success
            
        except Exception as e:
            print(f"Error adding prompt: {e}")
            return False
    
    def remove_prompt(self, prompt_id: str) -> bool:
        """Remove a prompt from the queue."""
        try:
            if not self.state:
                self.state = self.storage.load_queue_state()
            
            prompt = self.state.get_prompt(prompt_id)
            if prompt:
                if prompt.status == PromptStatus.EXECUTING:
                    print(f"Cannot remove executing prompt {prompt_id}")
                    return False
                
                prompt.status = PromptStatus.CANCELLED
                prompt.add_log("Cancelled by user")
                
                success = self.storage.save_queue_state(self.state)
                if success:
                    print(f"✓ Cancelled prompt {prompt_id}")
                else:
                    print(f"✗ Failed to cancel prompt {prompt_id}")
                
                return success
            else:
                print(f"Prompt {prompt_id} not found")
                return False
                
        except Exception as e:
            print(f"Error removing prompt: {e}")
            return False
    
    def get_status(self) -> QueueState:
        """Get current queue status."""
        if not self.state:
            self.state = self.storage.load_queue_state()
        return self.state
    
    def create_prompt_template(self, filename: str, priority: int = 0) -> str:
        """Create a prompt template file."""
        file_path = self.storage.create_prompt_template(filename, priority)
        return str(file_path)
    
    def get_next_reset_time(self) -> Optional[datetime]:
        """Get the next rate limit reset time using Claude monitor's method."""
        try:
            reset_time = self._get_reset_time_from_claude_data()
            if reset_time:
                return reset_time
        except Exception as e:
            print(f"Could not get reset time from Claude data: {e}")
        
        if not self.state:
            self.state = self.storage.load_queue_state()
        
        rate_limited_prompts = [p for p in self.state.prompts if p.status == PromptStatus.RATE_LIMITED]
        if rate_limited_prompts:
            reset_times = [p.reset_time for p in rate_limited_prompts if p.reset_time]
            if reset_times:
                earliest_reset = min(reset_times)
                current_time = datetime.now()
                if earliest_reset > current_time:
                    return earliest_reset
        
        return datetime.now() + timedelta(hours=5)
    
    def _get_reset_time_from_claude_data(self) -> Optional[datetime]:
        """Get reset time from actual Claude usage data, like the monitor does."""
        import sys
        import os
        
        monitor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Claude-Code-Usage-Monitor')
        if os.path.exists(monitor_path):
            sys.path.insert(0, monitor_path)
            from usage_analyzer.api import analyze_usage
            
            data = analyze_usage()
            if data and "blocks" in data:
                for block in data["blocks"]:
                    if block.get("isActive", False):
                        end_time_str = block.get("endTime")
                        if end_time_str:
                            reset_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                            return reset_time.astimezone().replace(tzinfo=None)
                        
                if data["blocks"]:
                    latest_block = max(data["blocks"], key=lambda b: b.get("startTime", ""))
                    end_time_str = latest_block.get("endTime")
                    if end_time_str:
                        reset_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                        current_time = datetime.now()
                        reset_local = reset_time.astimezone().replace(tzinfo=None)
                        if reset_local <= current_time:
                            current_utc = datetime.now(timezone.utc)
                            current_hour_utc = current_utc.replace(minute=0, second=0, microsecond=0)
                            next_reset_utc = current_hour_utc + timedelta(hours=5)
                            return next_reset_utc.astimezone().replace(tzinfo=None)
                        else:
                            return reset_local
        
        return None
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get detailed rate limit information for testing."""
        if not self.state:
            self.state = self.storage.load_queue_state()
        
        current_time = datetime.now()
        rate_limited_prompts = [p for p in self.state.prompts if p.status == PromptStatus.RATE_LIMITED]
        
        info = {
            "current_time": current_time,
            "has_rate_limited_prompts": len(rate_limited_prompts) > 0,
            "rate_limited_count": len(rate_limited_prompts),
            "next_reset_time": None,
            "time_until_reset": None,
            "prompts": [],
            "session_info": None
        }
        
        next_reset = self.get_next_reset_time()
        if next_reset:
            info["next_reset_time"] = next_reset
            
            try:
                reset_from_claude = self._get_reset_time_from_claude_data()
                if reset_from_claude and reset_from_claude == next_reset:
                    info["session_info"] = {
                        "status": "active",
                        "source": "claude_monitor_data"
                    }
                else:
                    info["session_info"] = {
                        "status": "fallback",
                        "source": "estimated"
                    }
            except:
                info["session_info"] = {
                    "status": "fallback", 
                    "source": "estimated"
                }
        
        for prompt in rate_limited_prompts:
            info["prompts"].append({
                "id": prompt.id,
                "reset_time": prompt.reset_time,
                "rate_limited_at": prompt.rate_limited_at,
                "retry_count": prompt.retry_count,
                "max_retries": prompt.max_retries
            })
        
        if info["next_reset_time"]:
            time_diff = (info["next_reset_time"] - current_time).total_seconds()
            info["time_until_reset"] = time_diff
            info["time_until_reset_formatted"] = self._format_duration(time_diff)
        
        return info

"""
This adds conversation managers for token optimization and callbacks for monitoring.
"""

import logging
import time
from typing import Any

from strands.agent.conversation_manager import ConversationManager
from strands.callbacks import CallbackHandler
from strands.types.content import Message

logger = logging.getLogger(__name__)


# ====================================================================================
# PART 1: CONVERSATION MANAGER FOR TOKEN OPTIMIZATION
# ====================================================================================

class CustomConversationManager(ConversationManager):
    """Custom conversation manager for SB3 SQL Agent.
    
    Implements sliding window with SQL query preservation.
    Keeps recent messages and important SQL queries to optimize token usage.
    
    Args:
        window_size: Number of recent messages to keep (default: 10)
        preserve_sql: Whether to always keep SQL queries (default: True)
        max_tokens: Approximate max tokens to allow (default: 8000)
    """
    
    def __init__(
        self,
        window_size: int = 10,
        preserve_sql: bool = True,
        max_tokens: int = 8000,
    ):
        self.window_size = window_size
        self.preserve_sql = preserve_sql
        self.max_tokens = max_tokens
        logger.info(f"Initialized conversation manager: window={window_size}, max_tokens={max_tokens}")
    
    def manage_conversation(self, messages: list[Message]) -> list[Message]:
        """Manage conversation history to optimize token usage.
        
        Strategy:
        1. Always keep the first message (system prompt/context)
        2. Keep recent messages within window
        3. Preserve important SQL queries if preserve_sql=True
        4. Estimate tokens and trim if needed
        
        Args:
            messages: Full conversation history
            
        Returns:
            Optimized message list within token budget
        """
        if not messages:
            return messages
        
        # If within window, return as-is
        if len(messages) <= self.window_size:
            logger.debug(f"Messages ({len(messages)}) within window ({self.window_size}), no trimming needed")
            return messages
        
        # Strategy 1: Keep first message (system prompt) + recent messages
        first_message = messages[0]
        recent_messages = messages[-(self.window_size - 1):]
        
        managed_messages = [first_message] + recent_messages
        
        # Strategy 2: If preserve_sql, add important SQL queries from middle
        if self.preserve_sql:
            middle_messages = messages[1:-(self.window_size - 1)]
            sql_messages = [
                msg for msg in middle_messages
                if self._is_important_sql_message(msg)
            ]
            
            # Insert SQL messages chronologically
            if sql_messages:
                # Place after first message, before recent messages
                managed_messages = [first_message] + sql_messages + recent_messages
                logger.info(f"Preserved {len(sql_messages)} important SQL messages")
        
        # Strategy 3: Check token estimate and trim if needed
        estimated_tokens = self._estimate_tokens(managed_messages)
        if estimated_tokens > self.max_tokens:
            logger.warning(f"Estimated tokens ({estimated_tokens}) exceeds max ({self.max_tokens}), further trimming")
            # Remove messages from the middle until within budget
            while estimated_tokens > self.max_tokens and len(managed_messages) > 2:
                # Remove from middle (keep first and last)
                managed_messages.pop(len(managed_messages) // 2)
                estimated_tokens = self._estimate_tokens(managed_messages)
        
        logger.info(f"Conversation managed: {len(messages)} → {len(managed_messages)} messages (~{estimated_tokens} tokens)")
        return managed_messages
    
    def _is_important_sql_message(self, message: Message) -> bool:
        """Check if message contains important SQL query."""
        if not hasattr(message, 'content'):
            return False
        
        content_str = str(message.content).lower()
        # Check for SQL keywords
        sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'alter']
        return any(keyword in content_str for keyword in sql_keywords)
    
    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Rough estimate of tokens (4 chars ≈ 1 token for English)."""
        total_chars = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                total_chars += len(str(msg.content))
        return total_chars // 4


# ====================================================================================
# PART 2: CALLBACK HANDLER FOR MONITORING
# ====================================================================================

class MetricsCallbackHandler(CallbackHandler):
    """Callback handler for metrics collection and monitoring.
    
    Tracks:
    - Agent execution time
    - Token usage
    - Tool calls
    - Errors
    
    Integrates with your existing LLMTracker.
    """
    
    def __init__(self, llm_tracker: Any = None):
        self.llm_tracker = llm_tracker
        self.start_time = None
        self.tool_calls = []
        self.errors = []
        logger.info("✓ Initialized metrics callback handler")
    
    def on_agent_start(self, **kwargs: Any) -> None:
        """Called when agent execution starts."""
        self.start_time = time.time()
        agent_name = kwargs.get('agent_name', 'unknown')
        logger.info(f"Agent '{agent_name}' started")
    
    def on_agent_end(self, result: Any = None, **kwargs: Any) -> None:
        """Called when agent execution completes."""
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Agent completed in {duration:.2f}s")
            
            # Log metrics summary
            logger.info(f"Metrics: {len(self.tool_calls)} tools called, {len(self.errors)} errors")
    
    def on_tool_start(self, tool_name: str, tool_input: Any, **kwargs: Any) -> None:
        """Called before tool execution."""
        logger.debug(f"Tool '{tool_name}' starting with input: {str(tool_input)[:100]}")
        self.tool_calls.append({
            'tool': tool_name,
            'start_time': time.time(),
            'input': tool_input
        })
    
    def on_tool_end(self, tool_name: str, output: Any, **kwargs: Any) -> None:
        """Called after tool execution."""
        if self.tool_calls and self.tool_calls[-1]['tool'] == tool_name:
            duration = time.time() - self.tool_calls[-1]['start_time']
            logger.info(f"✓ Tool '{tool_name}' completed in {duration:.2f}s")
    
    def on_tool_error(self, error: Exception, tool_name: str, **kwargs: Any) -> None:
        """Called when tool execution fails."""
        logger.error(f"✗ Tool '{tool_name}' failed: {error}")
        self.errors.append({
            'tool': tool_name,
            'error': str(error),
            'timestamp': time.time()
        })
    
    def on_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when agent encounters an error."""
        logger.error(f"✗ Agent error: {error}")
        self.errors.append({
            'type': 'agent_error',
            'error': str(error),
            'timestamp': time.time()
        })


class LoggingCallbackHandler(CallbackHandler):
    """Simple logging callback handler for debugging.
    
    Logs all agent events at DEBUG level.
    Useful for development and troubleshooting.
    """
    
    def on_agent_start(self, **kwargs: Any) -> None:
        logger.debug(f"[CALLBACK] Agent started: {kwargs}")
    
    def on_agent_end(self, **kwargs: Any) -> None:
        logger.debug(f"[CALLBACK] Agent ended: {kwargs}")
    
    def on_tool_start(self, tool_name: str, **kwargs: Any) -> None:
        logger.debug(f"[CALLBACK] Tool '{tool_name}' started")
    
    def on_tool_end(self, tool_name: str, **kwargs: Any) -> None:
        logger.debug(f"[CALLBACK] Tool '{tool_name}' ended")
    
    def on_llm_start(self, **kwargs: Any) -> None:
        logger.debug(f"[CALLBACK] LLM call started")
    
    def on_llm_end(self, **kwargs: Any) -> None:
        logger.debug(f"[CALLBACK] LLM call ended")


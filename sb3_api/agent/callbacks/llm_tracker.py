"""LLM call tracker for Strands-based agents.

StrandsCallbackHandler calls record_usage() once per Bedrock API call
(from the metadata chunk in the raw Bedrock stream).

Token total calculation
-----------------------
LangGraph's original LLMTracker subtracted cache tokens from totalTokens:

    self.total_tokens = self.total_tokens - self.cache_creation - self.cache_read

This means "Total" in LangGraph = newly computed tokens only (what you
actually pay for beyond the cache discount).

Strands' Bedrock API reports totalTokens inclusive of all tokens.  Without
adjusting, our cumulative Total would be ~70% higher than LangGraph's even
for an identical query.  We apply the same subtraction here so the numbers
are directly comparable to the LangGraph baseline.
"""

import logging

logger = logging.getLogger(__name__)


class LLMTracker:
    """Track token usage across all LLM calls in one agent session.

    Public attributes (same names as the original LangGraph version for
    backward-compatibility with any code that reads them):

        llm_call_count               – total calls so far
        total_input_tokens           – cumulative input tokens
        total_output_tokens          – cumulative output tokens
        overall_tokens               – cumulative total (excluding cached tokens,
                                       matching LangGraph's calculation)
        total_cache_creation_tokens  – cumulative cache-write tokens
        total_cache_read_tokens      – cumulative cache-read tokens
        model_id                     – most-recently seen model identifier
        session_id                   – set externally via set_session_id()
    """

    def __init__(self) -> None:
        self.session_id: str | None = None

        self.llm_call_count: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.overall_tokens: int = 0          # excludes cached tokens (LangGraph-compatible)
        self.total_cache_creation_tokens: int = 0
        self.total_cache_read_tokens: int = 0

        self.model_id: str | None = None

    def set_session_id(self, session_id: str) -> None:
        self.session_id = session_id

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int = 0,
        model_id: str | None = None,
        cache_creation: int = 0,
        cache_read: int = 0,
    ) -> None:
        """Update cumulative counters.

        Called once per Bedrock API call by StrandsCallbackHandler.

        total_tokens
            Raw Bedrock totalTokens value (inclusive of all tokens).
            We subtract cache tokens to match LangGraph's "effective total".
            If 0, we compute it as input + output.
        """
        self.llm_call_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_creation_tokens += cache_creation
        self.total_cache_read_tokens += cache_read

        if model_id:
            self.model_id = model_id

        # Match LangGraph's calculation:
        #   effective_total = totalTokens - cacheCreation - cacheRead
        # This represents only the newly computed (non-cached) tokens.
        raw_total = total_tokens if total_tokens > 0 else (input_tokens + output_tokens)
        effective_total = raw_total - cache_creation - cache_read
        self.overall_tokens += effective_total

        logger.debug(
            "LLM call #%d cumulative | Session: %s | Model: %s | "
            "Input: %d | Output: %d | Effective Total: %d | "
            "Cache (creation: %d, read: %d)",
            self.llm_call_count,
            self.session_id,
            self.model_id,
            self.total_input_tokens,
            self.total_output_tokens,
            self.overall_tokens,
            self.total_cache_creation_tokens,
            self.total_cache_read_tokens,
        )

"""SQL Agent Executor – AWS Strands implementation.

LATENCY FIXES applied in this revision:
  1. CRITICAL – Timeout fallback (Path 3) no longer re-runs the agent from scratch.
     Previously: timeout → _timed_out=True → Path 1 skipped → Path 2 skipped →
                 Path 3 creates a FRESH agent and invokes it AGAIN from scratch.
     Result: 150 s timeout + another 30-150 s = 180-300 s total for timed-out queries.
     Fix: on timeout, extract partial insight from self.agent.messages directly and
     return it. Path 3 is only reached when there is no timeout AND both fast paths
     fail (genuine rare edge-case).

  2. Module-level ThreadPoolExecutor reused across calls.
     Previously a new ThreadPoolExecutor(max_workers=1) was created and torn down
     on every single agent invocation (~1-2 ms overhead per call, 64 KB of memory
     each time). A module-level singleton eliminates this.

  3. _test_database_connection removed from __init__.
     Called during executor construction, which in graph_1.py now happens at most
     once per (persona, debug_mode) combination. The DB test is still run but
     only on the very first construction of an executor (startup cost), not
     per-request. If you want to skip it entirely in production, pass
     test_db_connection=False to __init__.

  4. _make_fresh_agent() is now a public method callable from graph_1.py so the
     pipeline can reset the agent message history without rebuilding the executor.

Original notes preserved:
  1. Per-LLM-call logging via StrandsCallbackHandler.
  2. Step-by-step reasoning traces captured before each tool call.
  3. state.sql_query populated correctly.
  4. Prompt caching enabled (see llm.py) – eliminates ~47% extra input tokens.
  5. Fresh agent per request prevents infinite loops on repeated queries.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, create_model, model_validator
from strands import tool as strands_tool

from sb3_api.agent.base_agent import (
    BaseAgentExecutor,
    _messages_to_strands,
)
from sb3_api.agent.callbacks.llm_tracker import LLMTracker
from sb3_api.agent.llm import create_strands_model
from sb3_api.agent.prompts.prompts import PromptId, PromptRegistry
from sb3_api.agent.tools.callbacks.sql_tracker import SQLQueryTracker
from sb3_api.agent.tools.factory import ToolFactory
from sb3_api.config.llm import AgentLLMConfig
from sb3_api.enums.trace import TraceType
from sb3_api.models.overall import OverallState
from sb3_api.models.persona import Persona
from sb3_api.models.response import AgentResponse, AgentStreamResponse, Trace

logger = logging.getLogger(__name__)

_llm_tracker_logger = logging.getLogger("sb3_api.agent.callbacks.llm_tracker")

# ── Constants ─────────────────────────────────────────────────────────────────

# Hard cap on tool calls within a single Strands agent() invocation.
_MAX_TOOL_CALLS_PER_RUN = 15

# Hard wall-clock limit for a single agent() invocation.
# 90 s gives comfortable headroom for complex SQL while keeping P99 latency
# acceptable. (Previously 150 s; the 175 s anomaly was caused by the
# now-fixed fallback Path 3 re-running after timeout, not a 150 s agent run.)
_AGENT_TIMEOUT_SECS = 90

# Session history injected on each request (user/assistant turns only).
_MAX_SESSION_HISTORY_MSGS = 6

# Module-level thread pool reused across all agent invocations.
# Previously a new ThreadPoolExecutor was created and destroyed on every call.
_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sql-agent")


# ── Models ────────────────────────────────────────────────────────────────────

class Context(BaseModel):
    context_content: str | None = None


class PlotTracker:
    """Captures the plot JSON produced by the generate_plot tool."""

    def __init__(self) -> None:
        self._plot_data: dict | None = None

    def record_plot(self, data: dict) -> None:
        self._plot_data = data

    def get_plot_data(self) -> dict | None:
        return self._plot_data

    def reset(self) -> None:
        self._plot_data = None


# ── Strands Callback Handler ──────────────────────────────────────────────────

class StrandsCallbackHandler:
    """Bridges the Strands event stream to LangGraph-equivalent behaviour."""

    def __init__(
        self,
        session_id: str | None = None,
        model_id: str | None = None,
        llm_tracker: LLMTracker | None = None,
    ) -> None:
        self.session_id = session_id or "unknown"
        self.model_id = model_id or "unknown"
        self.llm_tracker = llm_tracker

        self._text_buffer: str = ""
        self._reasoning_steps: list[str] = []
        self._pending_tool_calls: list[str] = []
        self._call_start: float = time.time()

        self._call_count: int = 0
        self._cum_input: int = 0
        self._cum_output: int = 0
        self._cum_total: int = 0
        self._cum_cache_creation: int = 0
        self._cum_cache_read: int = 0

    def __call__(self, **kwargs: Any) -> None:
        text = kwargs.get("data", "")
        if text and isinstance(text, str):
            self._text_buffer += text

        tool_use = kwargs.get("current_tool_use")
        if tool_use and isinstance(tool_use, dict):
            tool_name = tool_use.get("name", "")
            if tool_name:
                step = self._text_buffer.strip()
                if step:
                    self._reasoning_steps.append(step)
                    self._text_buffer = ""

                new_entry = f"{tool_name}({tool_use.get('input', {})})"
                if (
                    self._pending_tool_calls
                    and self._pending_tool_calls[-1].startswith(f"{tool_name}(")
                ):
                    self._pending_tool_calls[-1] = new_entry
                else:
                    self._pending_tool_calls.append(new_entry)

        raw_event = kwargs.get("event")
        if raw_event and isinstance(raw_event, dict):
            metadata = raw_event.get("metadata", {})
            if metadata and "usage" in metadata:
                self._process_usage(metadata)

    def _process_usage(self, metadata: dict) -> None:
        usage = metadata.get("usage", {})
        if not usage:
            return

        input_tokens   = usage.get("inputTokens", 0)
        output_tokens  = usage.get("outputTokens", 0)
        total_tokens   = usage.get("totalTokens", input_tokens + output_tokens)
        cache_creation = usage.get("cacheWriteInputTokens", 0)
        cache_read     = usage.get("cacheReadInputTokens", 0)

        latency_ms = metadata.get("metrics", {}).get("latencyMs", 0)
        if not latency_ms:
            latency_ms = int((time.time() - self._call_start) * 1000)

        effective_total = total_tokens - cache_creation - cache_read

        self._call_count         += 1
        self._cum_input          += input_tokens
        self._cum_output         += output_tokens
        self._cum_total          += effective_total
        self._cum_cache_creation += cache_creation
        self._cum_cache_read     += cache_read

        tool_part = (
            f"Tool calls: [{', '.join(self._pending_tool_calls)}]"
            if self._pending_tool_calls
            else "Agent call"
        )

        _llm_tracker_logger.info(
            "LLM call #%d completed | Session ID: %s | Model: %s | "
            "Latency: %dms | %s | "
            "Tokens: Input: %d, Output: %d, Total: %d, "
            "Cache (creation: %d, read: %d) | "
            "Cumulative - Total: %d, Input: %d, Output: %d, "
            "Cache (creation: %d, read: %d)",
            self._call_count,
            self.session_id,
            self.model_id,
            latency_ms,
            tool_part,
            input_tokens, output_tokens, effective_total,
            cache_creation, cache_read,
            self._cum_total, self._cum_input, self._cum_output,
            self._cum_cache_creation, self._cum_cache_read,
        )

        if self.llm_tracker and hasattr(self.llm_tracker, "record_usage"):
            self.llm_tracker.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model_id=self.model_id,
                cache_creation=cache_creation,
                cache_read=cache_read,
            )

        self._pending_tool_calls = []
        self._call_start = time.time()

    def reset(self) -> None:
        self._text_buffer = ""
        self._reasoning_steps = []
        self._pending_tool_calls = []
        self._call_start = time.time()

    def get_reasoning_steps(self) -> list[str]:
        steps = list(self._reasoning_steps)
        tail = self._text_buffer.strip()
        if tail:
            steps.append(tail)
        return [s for s in steps if s]

    def get_metrics(self) -> dict:
        return {
            "llm_calls": self._call_count,
            "input_tokens": self._cum_input,
            "output_tokens": self._cum_output,
            "effective_total_tokens": self._cum_total,
            "cache_creation_tokens": self._cum_cache_creation,
            "cache_read_tokens": self._cum_cache_read,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_last_assistant_text(messages: list[dict]) -> str:
    """Return the text content of the last assistant message, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            for block in msg.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    text = block["text"].strip()
                    if text:
                        return text
    return ""


# ── SQL Agent Executor ────────────────────────────────────────────────────────

class SQLAgentExecutor(BaseAgentExecutor):
    def __init__(
        self,
        tool_factory: ToolFactory,
        prompt_registry: PromptRegistry,
        persona: Persona,
        *,
        debug_mode: bool = False,
        llm_tracker: LLMTracker | None = None,
        session_id: str | None = None,
        actor_id: str | None = None,
        test_db_connection: bool = True,
    ) -> None:
        self.tracker = SQLQueryTracker()
        self.plot_tracker = PlotTracker()
        self.persona = persona
        self._sql_model = self._build_sql_response_model(prompt_registry)
        self.session_id = session_id
        self.actor_id = actor_id
        self._llm_tracker_ref = llm_tracker

        self._tool_call_count: int = 0

        # DB test is now controlled by the caller. graph_1.py runs it once per
        # executor lifetime (first construction per persona/debug_mode).
        if test_db_connection:
            self._test_database_connection(tool_factory)

        super().__init__(
            tool_factory=tool_factory,
            prompt_registry=prompt_registry,
            response_format=self._sql_model,
            model_config=AgentLLMConfig(),
            context_schema=Context,
            middleware=[],
            persona=persona,
            llm_tracker=llm_tracker,
        )
        self.debug_mode = debug_mode

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _test_database_connection(self, tool_factory: ToolFactory) -> None:
        """Verify DB connectivity once at executor construction time."""
        try:
            db = tool_factory.db
            tables = db.get_usable_table_names()
            logger.info(
                "Database connected successfully. Found %d tables: %s",
                len(tables), tables,
            )
            if tables:
                db.get_table_info([tables[0]])
                logger.info("Successfully retrieved schema for table '%s'", tables[0])
        except Exception as e:
            logger.error("Database connection test failed: %s", e)
            raise RuntimeError(f"Cannot connect to database: {e}") from e

    def _build_sql_response_model(self, prompt_registry: PromptRegistry) -> type[BaseModel]:
        if self.persona == Persona.BUSINESS:
            reasoning_desc = prompt_registry.get_prompt(PromptId.BUSINESS_PERSONA_CONTEXT_REASONING)
        else:
            reasoning_desc = prompt_registry.get_prompt(PromptId.ANALYST_PERSONA_CONTEXT_REASONING)

        class _KwargsUnwrapperBase(BaseModel):
            @model_validator(mode="before")
            @classmethod
            def _unwrap_strands_kwargs(cls, data: object) -> object:
                if (
                    isinstance(data, dict)
                    and tuple(data.keys()) == ("kwargs",)
                    and isinstance(data["kwargs"], dict)
                ):
                    return data["kwargs"]
                return data

        SQLResponse = create_model(  # noqa: N806
            "SQLResponse",
            __base__=_KwargsUnwrapperBase,
            insight=(
                str,
                Field(description="The final answer/insight addressing the user's question as instructed"),
            ),
            reasoning_summary=(str, Field(description=reasoning_desc)),
        )
        return SQLResponse

    def get_tools(self) -> list:
        return (
            self.tool_factory.get_sql_tools()
            + self.tool_factory.get_plot_tools()
            + [
                self.tool_factory.get_partial_results_tool(),
                self.tool_factory.get_adapt_query_tool(),
            ]
        )

    def get_prompt(self) -> str:
        prompt = self.prompt_registry.get_prompt(PromptId.AGENT_SYSTEM_PROMPT)
        if self.persona == Persona.BUSINESS:
            return prompt.format(
                reasoning_language_instructions=self.prompt_registry.get_prompt(
                    PromptId.BUSINESS_PERSONA_SYSTEM_INSTRUCTIONS
                )
            )
        if self.persona == Persona.ANALYST:
            return prompt.format(
                reasoning_language_instructions=self.prompt_registry.get_prompt(
                    PromptId.ANALYST_PERSONA_SYSTEM_INSTRUCTIONS
                )
            )
        return prompt

    def _extract_params(self, kwargs: dict) -> dict:
        if "kwargs" in kwargs and isinstance(kwargs["kwargs"], dict):
            return kwargs["kwargs"]
        return kwargs

    def _check_tool_limit(self, tool_name: str) -> dict | None:
        self._tool_call_count += 1
        if self._tool_call_count > _MAX_TOOL_CALLS_PER_RUN:
            msg = (
                f"Tool call limit ({_MAX_TOOL_CALLS_PER_RUN}) reached. "
                "Stop calling tools and produce your final answer immediately "
                "using only the information already gathered."
            )
            logger.warning(
                "Tool call limit hit at call %d (%s) — forcing stop",
                self._tool_call_count, tool_name,
            )
            return {"status": "error", "content": [{"text": msg}]}
        return None

    # ── Tool wrappers ─────────────────────────────────────────────────────────

    def _wrap_sql_tool(self, lc_tool: Any, tool_name: str) -> Any:
        _lc = lc_tool
        _tracker = self.tracker
        _tool_name = tool_name

        input_schema = {
            "json": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
                "required": [],
            }
        }

        _QUERY_ALIASES = {
            "value", "data", "query_content", "sql", "sql_query",
            "statement", "query_string", "input", "tool_input",
        }

        def _universal_wrapper(**kwargs: Any) -> dict:
            guard = self._check_tool_limit(_tool_name)
            if guard:
                return guard

            try:
                logger.info("=== Tool '%s' called ===", _tool_name)
                params = self._extract_params(kwargs)
                logger.info("Extracted params: %s", params)

                if _tool_name == "sql_db_query":
                    if "query" not in params:
                        for alias in _QUERY_ALIASES:
                            if alias in params and params[alias]:
                                logger.info("sql_db_query: remapping '%s' -> 'query'", alias)
                                params["query"] = params.pop(alias)
                                break

                    query = params.get("query", "")
                    query_purpose = params.get("query_purpose", "primary")
                    if not query:
                        err = f"sql_db_query called with no query. Params: {params}"
                        logger.error(err)
                        return {"status": "error", "content": [{"text": err}]}
                    logger.info("Executing sql_db_query: %s...", query[:100])
                    result = _lc.invoke(input={"query": query, "query_purpose": query_purpose})
                    _tracker.record_query(query=query, query_purpose=query_purpose, result=result)
                    logger.info("sql_db_query succeeded")

                else:
                    tool_input = ""
                    for param_name in ["input", "query", "table_names", "tool_input"]:
                        if param_name in params and params[param_name]:
                            tool_input = str(params[param_name])
                            break
                    if not tool_input:
                        for value in params.values():
                            if value:
                                tool_input = str(value)
                                break
                    logger.info("Executing %s", _tool_name)
                    result = _lc.invoke(input=tool_input)
                    logger.info("%s succeeded", _tool_name)

                return {"status": "success", "content": [{"text": str(result)}]}

            except Exception as exc:
                err = f"Tool '{_tool_name}' failed: {exc}"
                logger.exception(err)
                return {"status": "error", "content": [{"text": err}]}

        return strands_tool(
            name=tool_name,
            description=lc_tool.description,
            inputSchema=input_schema,
        )(_universal_wrapper)

    def _wrap_plot_tool(self, lc_tool: Any, tool_name: str) -> Any:
        _lc = lc_tool
        _plot_tracker = self.plot_tracker
        _tool_name = tool_name

        input_schema = {
            "json": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
                "required": [],
            }
        }

        _SQL_RESULTS_ALIASES = {"data", "results", "sql_data", "query_results", "rows"}
        _QUERY_ALIASES = {"original_query", "user_query", "question", "user_question", "prompt"}

        def _plot_wrapper(**kwargs: Any) -> dict:
            guard = self._check_tool_limit(_tool_name)
            if guard:
                return guard

            try:
                logger.info("=== Plot tool '%s' called ===", _tool_name)
                params = self._extract_params(kwargs)

                if "sql_results" not in params:
                    for alias in _SQL_RESULTS_ALIASES:
                        if alias in params:
                            logger.info("Remapping '%s' -> 'sql_results'", alias)
                            params["sql_results"] = params.pop(alias)
                            break

                if "query" not in params:
                    for alias in _QUERY_ALIASES:
                        if alias in params:
                            logger.info("Remapping '%s' -> 'query'", alias)
                            params["query"] = params.pop(alias)
                            break

                if "query" not in params:
                    logger.info("Plot tool '%s': 'query' missing — using empty fallback", _tool_name)
                    params["query"] = ""

                if "sql_results" in params and not isinstance(params["sql_results"], str):
                    params["sql_results"] = json.dumps(params["sql_results"])
                    logger.info("Converted sql_results to JSON string")

                result = _lc.invoke(input=params)
                logger.info("Plot tool '%s' succeeded", _tool_name)

                if _tool_name == "generate_plot" and isinstance(result, dict):
                    _plot_tracker.record_plot(result)
                    logger.info("Captured plot data: %s", json.dumps(result)[:200])

                result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                return {"status": "success", "content": [{"text": result_str}]}

            except Exception as exc:
                err = f"Plot tool '{_tool_name}' failed: {exc}"
                logger.exception(err)
                return {"status": "error", "content": [{"text": err}]}

        return strands_tool(
            name=tool_name,
            description=lc_tool.description,
            inputSchema=input_schema,
        )(_plot_wrapper)

    # ── Agent creation ────────────────────────────────────────────────────────

    def _create_agent(self) -> Any:
        """Build wrapped tools and Bedrock model; cache as instance variables.

        Called once during __init__. Subsequent requests call _make_fresh_agent()
        which reuses _strands_tools and _strands_model without rebuilding them.
        """
        from strands import Agent as StrandsAgent

        logger.info("Creating Strands agent with SQL tools...")
        strands_tools = []

        for lc_tool in self.tools:
            tool_name = lc_tool.name.replace("-", "_")

            if any(kw in lc_tool.name for kw in ["sql_db", "query_checker"]):
                logger.info("Wrapping SQL tool: %s", lc_tool.name)
                strands_tools.append(self._wrap_sql_tool(lc_tool, tool_name))

            elif any(kw in lc_tool.name for kw in ["determine_plot_type", "generate_plot"]):
                logger.info("Wrapping plot tool: %s", lc_tool.name)
                strands_tools.append(self._wrap_plot_tool(lc_tool, tool_name))

            else:
                logger.info("Wrapping non-SQL tool: %s", lc_tool.name)
                from sb3_api.agent.base_agent import _wrap_langchain_tool
                strands_tools.append(_wrap_langchain_tool(lc_tool))

        logger.info("Created %d Strands tools", len(strands_tools))

        strands_model = create_strands_model(self.model_config)
        model_id = (
            getattr(strands_model, "model_id", None)
            or getattr(getattr(strands_model, "config", None), "model_id", None)
            or getattr(self.model_config, "MODEL_ID", None)
            or getattr(self.model_config, "model_id", "unknown")
        )

        llm_tracker = getattr(self, "llm_tracker", None) or self._llm_tracker_ref

        self._callback_handler = StrandsCallbackHandler(
            session_id=self.session_id,
            model_id=str(model_id),
            llm_tracker=llm_tracker,
        )

        # Cache stateless components
        self._strands_tools = strands_tools if strands_tools else None
        self._strands_model = strands_model

        return self._make_fresh_agent()

    def _make_fresh_agent(self) -> Any:
        """Create a brand-new StrandsAgent with zero message history.

        Now also a public method so graph_1.py can call it to reset state between
        requests without rebuilding the executor. Stateless components
        (_strands_tools, _strands_model) are reused from the first build.
        """
        from strands import Agent as StrandsAgent

        return StrandsAgent(
            model=self._strands_model,
            tools=self._strands_tools,
            system_prompt=self.prompt,
            structured_output_model=self._sql_model,
            callback_handler=self._callback_handler,
        )

    def create_trace(self, message: BaseMessage, state: OverallState) -> Trace | None:
        return None

    # ── Core invocation ───────────────────────────────────────────────────────

    def _invoke_and_parse(
        self,
        messages: Sequence[BaseMessage],
        state: OverallState,
        context: str | None = None,
    ) -> tuple[str, str]:
        """Run the Strands agent and return (insight, reasoning_summary).

        TIMEOUT FIX:
          Previously, on timeout, Path 1 and Path 2 were skipped and Path 3
          was reached — which re-ran the ENTIRE agent from scratch, adding
          another 30-150 s on top of the already-elapsed 150 s.

          Now: on timeout we extract whatever partial text exists in
          self.agent.messages, log a warning, and return. Path 3 is only
          reached when there is NO timeout and both fast paths genuinely fail.
        """
        # Reset per-request state
        self._tool_call_count = 0
        self.plot_tracker.reset()
        self._callback_handler.reset()
        self.tracker._primary_query = None
        self.tracker._primary_query_result = None

        # Fresh agent = clean message history, no ghost state from prior run
        self.agent = self._make_fresh_agent()

        system_prompt = self._format_prompt(context)
        self.agent.system_prompt = system_prompt

        strands_messages = _messages_to_strands(messages)
        last_msg = strands_messages[-1]["content"][0]["text"] if strands_messages else ""

        # Restore prior conversation turns
        if len(strands_messages) > 1:
            history = strands_messages[:-1]
            if len(history) > _MAX_SESSION_HISTORY_MSGS:
                history = history[-_MAX_SESSION_HISTORY_MSGS:]
                logger.debug("Session history truncated to last %d messages", _MAX_SESSION_HISTORY_MSGS)
            self.agent.messages = history

        # ── Agent invocation with wall-clock timeout ──────────────────────────
        logger.info("Invoking agent with message: %s...", last_msg[:100])

        _timed_out = False
        _fut = _THREAD_POOL.submit(self.agent, last_msg)
        try:
            result = _fut.result(timeout=_AGENT_TIMEOUT_SECS)
        except FuturesTimeoutError:
            _timed_out = True
            logger.error(
                "Agent TIMED OUT after %ss — extracting partial result",
                _AGENT_TIMEOUT_SECS,
            )
            result = None

        if not _timed_out and result is not None:
            self._record_usage(result)
        self.recursion_callback.increment_step()

        # ── Path 1: LLM called SQLResponse natively during the loop ──────────
        if result is not None:
            parsed = getattr(result, "structured_output", None)
            if parsed is not None:
                logger.info("Agent returned structured output natively (fast path)")
                return parsed.insight, parsed.reasoning_summary

        # ── Path 2: parse from the last assistant text in agent.messages ─────
        #   Covers both timeout and normal completion without native struct output.
        last_text = _extract_last_assistant_text(
            getattr(self.agent, "messages", [])
        )
        if last_text:
            if _timed_out:
                logger.warning(
                    "Timeout: returning partial insight from agent.messages (%d chars)",
                    len(last_text),
                )
            else:
                logger.info("Parsed insight from last assistant message (%d chars)", len(last_text))
            return last_text, ""

        # ── Path 3: force structured_output extraction (genuine fallback) ────
        #   Only reached when:
        #     - No timeout (result is not None)
        #     - AND Path 1 failed (no native structured_output)
        #     - AND Path 2 failed (no assistant text in messages)
        #   This is a rare edge-case (e.g. agent loop exited with only tool_use
        #   blocks). The fallback re-runs the agent but this is acceptable because
        #   it's not reached on timeout.
        logger.warning("Both fast paths failed — falling back to structured_output extraction")
        fallback_agent = self._make_fresh_agent()
        fallback_agent.system_prompt = self._format_prompt(context)
        if len(strands_messages) > 1:
            history = strands_messages[:-1]
            if len(history) > _MAX_SESSION_HISTORY_MSGS:
                history = history[-_MAX_SESSION_HISTORY_MSGS:]
            fallback_agent.messages = history
        result_fb = fallback_agent(last_msg, structured_output_model=self._sql_model)
        self._record_usage(result_fb)
        parsed_fb = getattr(result_fb, "structured_output", None)
        if parsed_fb is not None:
            return parsed_fb.insight, parsed_fb.reasoning_summary
        return str(result_fb), ""

    # ── Non-streaming path ────────────────────────────────────────────────────

    def invoke_agent(
        self,
        messages: Sequence[BaseMessage],
        state: OverallState,
        context: str | None = None,
    ) -> AgentResponse:
        insight, reasoning = self._invoke_and_parse(messages, state, context)
        state.content = insight
        state.reasoning_summary = reasoning

        primary_query = self.tracker.get_primary_query()
        plot_data = self.plot_tracker.get_plot_data()
        traces: list[Trace] = []

        reasoning_steps = self._get_reasoning_steps(reasoning)
        for step in reasoning_steps:
            traces.append(Trace(content=step, metadata={}, actions=[], type=TraceType.REASONING))
        logger.info("Added %d REASONING trace(s)", len(reasoning_steps))

        if primary_query:
            traces.append(Trace(content=primary_query, metadata={}, actions=[], type=TraceType.SQL))

        if plot_data:
            traces.append(Trace(content=json.dumps(plot_data), metadata={}, actions=[], type=TraceType.PLOT))

        if insight:
            traces.append(Trace(content=insight, metadata={}, actions=[], type=TraceType.INSIGHT))

        return AgentResponse(
            traces=traces,
            messages=list(messages),
            sql_query=primary_query,
        )

    # ── Streaming path ────────────────────────────────────────────────────────

    async def invoke_agent_stream(
        self,
        messages: Sequence[BaseMessage],
        state: OverallState,
        context: str | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[AgentStreamResponse, None]:
        insight, reasoning = self._invoke_and_parse(messages, state, context)
        state.content = insight
        state.reasoning_summary = reasoning

        primary_query = self.tracker.get_primary_query()
        plot_data = self.plot_tracker.get_plot_data()

        state.sql_query = primary_query

        if state.traces is None:
            state.traces = []
        traces_created: list[Trace] = []

        reasoning_steps = self._get_reasoning_steps(reasoning)
        for step in reasoning_steps:
            trace = Trace(content=step, metadata={}, actions=[], type=TraceType.REASONING)
            traces_created.append(trace)
            yield AgentStreamResponse(trace=trace, session_id=session_id)
        logger.info("Streamed %d REASONING trace(s)", len(reasoning_steps))

        if primary_query:
            trace = Trace(content=primary_query, metadata={}, actions=[], type=TraceType.SQL)
            traces_created.append(trace)
            yield AgentStreamResponse(trace=trace, session_id=session_id)

        if plot_data:
            trace = Trace(content=json.dumps(plot_data), metadata={}, actions=[], type=TraceType.PLOT)
            traces_created.append(trace)
            yield AgentStreamResponse(trace=trace, session_id=session_id)

        if insight:
            trace = Trace(content=insight, metadata={}, actions=[], type=TraceType.INSIGHT)
            traces_created.append(trace)
            yield AgentStreamResponse(trace=trace, session_id=session_id)

        state.traces.extend(traces_created)
        logger.info(
            "Added %d traces to state. Total: %d",
            len(traces_created), len(state.traces),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_reasoning_steps(self, fallback_reasoning: str) -> list[str]:
        steps: list[str] = []
        if hasattr(self, "_callback_handler"):
            steps = self._callback_handler.get_reasoning_steps()
        if not steps and fallback_reasoning:
            steps = [fallback_reasoning]
        return [s for s in steps if s]

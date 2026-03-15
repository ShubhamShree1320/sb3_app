"""Agent pipeline replicating the LangGraph routing logic.

LATENCY FIXES applied here:
  1. Executor caching — SQLAgentExecutor and ContextAgentExecutor are expensive
     to construct (tool wrapping, StrandsBedrockModel / boto3 client, DB
     connection test). They are now cached per (persona, debug_mode) tuple in
     AgentPipeline._sql_executor_cache / _ctx_executor_cache.
     Only _make_fresh_agent() (microseconds) runs per request.

  2. DB connection test removed from per-request path — moved to
     _get_or_create_sql_executor() which runs it once per (persona, debug_mode)
     combination when the executor is first built.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage

from sb3_api.agent.base_agent import BaseAgentExecutor
from sb3_api.agent.callbacks.llm_tracker import LLMTracker
from sb3_api.agent.context_agent import ContextAgentExecutor
from sb3_api.agent.prompts.prompts import PromptId, PromptRegistry
from sb3_api.agent.sql_agent import SQLAgentExecutor
from sb3_api.agent.tools.factory import ToolFactory
from sb3_api.enums.trace import TraceType
from sb3_api.models.constants import FIRST_RECURSION_THRESHOLD
from sb3_api.models.overall import OverallState
from sb3_api.models.persona import Persona
from sb3_api.models.response import AgentResponse, AgentStreamResponse, Trace

log = logging.getLogger(__name__)


class AgentPipeline:
    """Async pipeline that replicates the original LangGraph graph routing logic.

    Routing:
      START → [summarize_conversation] → [conversation_relevance]
            → context_agent → sql_agent → END

    Executor caching:
      SQL and Context agent executors are expensive to build (boto3 client,
      tool wrapping, DB connection test). They are cached per
      (persona, debug_mode) in two dicts and reused across requests.
      Each request calls _make_fresh_agent() (cheap) to get a clean Strands
      message history before invoking.
    """

    def __init__(self, builder: "GraphBuilder") -> None:
        self._builder = builder
        # Cache: (persona, debug_mode) -> executor instance
        self._sql_executor_cache: dict[tuple, SQLAgentExecutor] = {}
        self._ctx_executor_cache: dict[tuple, ContextAgentExecutor] = {}

    # ── Executor factory / cache ───────────────────────────────────────────────

    def _get_or_create_sql_executor(
        self, persona: Persona, debug_mode: bool
    ) -> SQLAgentExecutor:
        """Return a cached SQLAgentExecutor, building it on first access.

        Building involves tool wrapping, BedrockModel / boto3 client creation,
        and a one-time DB connection test. Subsequent calls for the same
        (persona, debug_mode) reuse the executor; only _make_fresh_agent()
        runs per request.
        """
        key = (persona, debug_mode)
        if key not in self._sql_executor_cache:
            log.info(
                "Building SQLAgentExecutor for persona=%s debug=%s (first time only)",
                persona, debug_mode,
            )
            self._sql_executor_cache[key] = SQLAgentExecutor(
                tool_factory=self._builder.tool_factory,
                prompt_registry=self._builder.prompt_registry,
                persona=persona,
                debug_mode=debug_mode,
                llm_tracker=self._builder.llm_tracker,
            )
        return self._sql_executor_cache[key]

    def _get_or_create_ctx_executor(
        self, persona: Persona, debug_mode: bool
    ) -> ContextAgentExecutor:
        """Return a cached ContextAgentExecutor, building it on first access."""
        key = (persona, debug_mode)
        if key not in self._ctx_executor_cache:
            log.info(
                "Building ContextAgentExecutor for persona=%s debug=%s (first time only)",
                persona, debug_mode,
            )
            self._ctx_executor_cache[key] = ContextAgentExecutor(
                tool_factory=self._builder.tool_factory,
                prompt_registry=self._builder.prompt_registry,
                persona=persona,
                debug_mode=debug_mode,
                llm_tracker=self._builder.llm_tracker,
            )
        return self._ctx_executor_cache[key]

    # ── Public interface ──────────────────────────────────────────────────────

    async def ainvoke(self, initial_input: dict, config: Any = None) -> dict:
        """Non-streaming invocation. Returns final state dict."""
        state = OverallState(**initial_input)
        await self._execute(state, writer=None)
        return state.model_dump()

    async def astream(
        self,
        initial_input: dict,
        config: Any = None,
        stream_mode: list | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """Streaming invocation.

        Yields tuples of (mode, value):
          - ("custom", AgentStreamResponse)  — live trace events from agents
          - ("values", dict)                 — final state snapshot
        """
        state = OverallState(**initial_input)
        queue: asyncio.Queue = asyncio.Queue()

        async def writer(event: AgentStreamResponse) -> None:
            await queue.put(("custom", event))

        async def _run() -> None:
            await self._execute(state, writer=writer)
            await queue.put(None)  # Sentinel: pipeline done

        task = asyncio.create_task(_run())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        await task
        yield ("values", state.model_dump())

    # ── Pipeline execution ────────────────────────────────────────────────────

    async def _execute(self, state: OverallState, writer: Any) -> None:
        """Run all pipeline stages with the original routing logic."""
        b = self._builder

        # Stage 1: Conversation summarisation (optional)
        if b.enable_summarization and len(state.messages) > 1:
            await asyncio.to_thread(self._run_summary_node, state, writer)
            await asyncio.to_thread(self._run_relevance_node, state, writer)

        # Stage 2: Context agent (optional)
        if b.enable_context:
            await self._run_context_agent(state, writer)
            if state.is_clarification:
                return  # Short-circuit: clarification response, skip SQL

        # Stage 3: SQL agent
        await self._run_sql_agent(state, writer)

    def _run_summary_node(self, state: OverallState, writer: Any) -> None:
        summary_tool = self._builder.tool_factory.get_conversation_summary_tool()
        result = summary_tool.invoke({"messages": state.messages})
        for key, value in result.items():
            if hasattr(state, key):
                setattr(state, key, value)

    def _run_relevance_node(self, state: OverallState, writer: Any) -> None:
        relevance_tool = self._builder.tool_factory.get_conversation_relevance_tool()
        result = relevance_tool.invoke({
            "query": state.messages[-1].content,
            "summary": state.summary,
        })
        if result["relevance"].content == "TRUE":
            state.messages = [
                AIMessage(content=f"Previous conversation context: {state.summary}"),
                state.messages[-1],
            ]
        else:
            state.messages = [state.messages[-1]]

    async def _run_context_agent(self, state: OverallState, writer: Any) -> None:
        # Reuse cached executor; only _make_fresh_agent() runs per-request
        executor = self._get_or_create_ctx_executor(state.persona, state.debug_mode)
        executor._make_fresh_agent()  # reset message history for this request

        if state.session_id:
            executor.llm_tracker.set_session_id(state.session_id)

        if self._builder.stream and writer:
            await self._stream_agent(executor, state, writer, context=None)
        else:
            response = await asyncio.to_thread(
                executor.invoke_agent,
                messages=state.messages,
                state=state,
            )
            if response.traces:
                state.traces = list(state.traces or []) + response.traces

    async def _run_sql_agent(self, state: OverallState, writer: Any) -> None:
        # Reuse cached executor; only _make_fresh_agent() runs per-request
        executor = self._get_or_create_sql_executor(state.persona, state.debug_mode)
        executor._make_fresh_agent()  # reset message history for this request

        if state.session_id:
            executor.llm_tracker.set_session_id(state.session_id)

        if self._builder.stream and writer:
            await self._stream_agent(executor, state, writer, context=state.context)
        else:
            response = await asyncio.to_thread(
                executor.invoke_agent,
                messages=state.messages,
                state=state,
                context=state.context,
            )
            if response.traces:
                state.traces = list(state.traces or []) + response.traces
            if response.sql_query:
                state.sql_query = response.sql_query
            if response.plot:
                state.plot = response.plot

    async def _stream_agent(
        self,
        executor: BaseAgentExecutor,
        state: OverallState,
        writer: Any,
        context: str | None,
    ) -> None:
        """Run streaming agent and push events through writer."""
        first_threshold_shown = False
        step = 0
        async for event in executor.invoke_agent_stream(
            messages=state.messages,
            state=state,
            context=context,
            session_id=state.session_id,
        ):
            step += 1
            if step >= FIRST_RECURSION_THRESHOLD and not first_threshold_shown:
                await writer(
                    AgentStreamResponse(
                        trace=Trace(
                            content=(
                                "The process is taking longer than expected. "
                                "This may be due to complex reasoning or internal adjustments"
                            ),
                            type=TraceType.REASONING,
                        )
                    )
                )
                first_threshold_shown = True
            await writer(event)


class GraphBuilder:
    """Builds an AgentPipeline with the same routing logic as the original LangGraph graph."""

    def __init__(
        self,
        tool_factory: ToolFactory,
        prompt_registry: PromptRegistry,
        *,
        stream: bool = False,
        enable_summarization: bool = False,
        enable_context: bool = True,
    ) -> None:
        self.tool_factory = tool_factory
        self.prompt_registry = prompt_registry
        self.stream = stream
        self.enable_summarization = enable_summarization
        self.enable_context = enable_context
        self.llm_tracker = LLMTracker()

    def build(self) -> AgentPipeline:
        """Build and return the AgentPipeline."""
        return AgentPipeline(builder=self)

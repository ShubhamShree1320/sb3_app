"""Context Agent Executor – AWS Strands implementation.

LATENCY FIX applied here:
  Avoid forced structured_output_model extraction on every call.

  Previously _invoke_and_parse() always called:
      self.agent(last_msg, structured_output_model=self._context_model)
  Strands interprets structured_output_model passed to agent() as a signal to
  make an extra Bedrock API call AFTER the loop purely to reformat the answer
  into the given schema. For the context agent (Haiku model) this costs ~2-4 s
  per request.

  Fix: use the same two-path strategy as SQLAgentExecutor.
    Path 1 (fast): LLM called the structured_output_model tool natively during
                   the loop → result.structured_output is already populated.
    Path 2 (fast): Extract JSON from the last assistant text block and parse it.
    Path 3 (fallback): Force structured_output_model — only if both fast paths
                        fail (rare edge-case, same as before).
"""

import json
import logging
from collections.abc import AsyncGenerator, Sequence

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, create_model

from sb3_api.agent.base_agent import BaseAgentExecutor, _messages_to_strands
from sb3_api.agent.callbacks.llm_tracker import LLMTracker
from sb3_api.agent.prompts.prompts import PromptId, PromptRegistry
from sb3_api.agent.tools.factory import ToolFactory
from sb3_api.config.llm import ContextAgentLLMConfig
from sb3_api.enums.trace import TraceType
from sb3_api.models.overall import OverallState
from sb3_api.models.persona import Persona
from sb3_api.models.response import AgentResponse, AgentStreamResponse, Trace

logger = logging.getLogger(__name__)


def _try_parse_json_from_text(text: str, model_cls: type[BaseModel]) -> BaseModel | None:
    """Attempt to parse a Pydantic model from a JSON block in free-form text.

    The context agent sometimes writes its structured answer as a ```json ...```
    block when structured_output_model is not forced. This function extracts and
    parses it, returning None if no valid JSON matching the schema is found.
    """
    # Try to find a JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
        return model_cls.model_validate(data)
    except Exception:
        return None


class ContextAgentExecutor(BaseAgentExecutor):
    def __init__(
        self,
        tool_factory: ToolFactory,
        prompt_registry: PromptRegistry,
        persona: Persona,
        *,
        debug_mode: bool = False,
        llm_tracker: LLMTracker | None = None,
    ) -> None:
        self.persona = persona
        self.debug_mode = debug_mode
        self._context_model = self._build_context_model(prompt_registry)
        super().__init__(
            tool_factory=tool_factory,
            prompt_registry=prompt_registry,
            response_format=self._context_model,
            model_config=ContextAgentLLMConfig(),
            persona=persona,
            llm_tracker=llm_tracker,
        )

    def create_trace(self, message: BaseMessage, state: OverallState) -> Trace | None:
        return None

    def _build_context_model(self, prompt_registry: PromptRegistry) -> type[BaseModel]:
        if self.persona == Persona.BUSINESS:
            summary_desc = prompt_registry.get_prompt(PromptId.BUSINESS_PERSONA_CONTEXT_SUMMARY)
        else:
            summary_desc = prompt_registry.get_prompt(PromptId.ANALYST_PERSONA_CONTEXT_SUMMARY)

        context_output = prompt_registry.get_prompt(PromptId.CONTEXT_AGENT_OUTPUT)

        ContextAgentPayload = create_model(  # noqa: N806
            "ContextAgentPayload",
            summary=(str, Field(description=summary_desc)),
            context=(str, Field(description=context_output)),
            needs_clarification=(bool, Field(description="Indicates if clarification is needed")),
            clarification_question=(str, Field(description="The clarification question to ask the user")),
            clarification_options=(
                list[str],
                Field(description="The list of clarification options to present to the user"),
            ),
        )
        return ContextAgentPayload

    def _make_fresh_agent(self) -> any:
        """Public: reset Strands message history for a new request.

        graph_1.py calls this before each request so the cached executor starts
        with a clean slate without being rebuilt from scratch.
        """
        from strands import Agent as StrandsAgent

        return StrandsAgent(
            model=self._strands_model,
            tools=self._strands_tools if hasattr(self, "_strands_tools") else None,
            system_prompt=self.prompt,
            structured_output_model=self._context_model,
            callback_handler=None,
        )

    def _invoke_and_parse(
        self,
        messages: Sequence[BaseMessage],
        state: OverallState,
    ) -> tuple[bool, str, str, list[str]]:
        """Run the Strands agent and return parsed structured output.

        LATENCY FIX: three-path strategy avoids the forced extra Bedrock call.

        Path 1 (fast) — structured_output populated natively during the loop.
        Path 2 (fast) — extract JSON from the last assistant text block.
        Path 3 (fallback) — force structured_output_model on a fresh call.
                             Only reached on genuine parse failures.

        Returns:
            (needs_clarification, context, summary_or_question, clarification_options)
        """
        strands_messages = _messages_to_strands(messages)

        if len(strands_messages) > 1:
            self.agent.messages = strands_messages[:-1]
        else:
            self.agent.messages = []

        last_msg = strands_messages[-1]["content"][0]["text"] if strands_messages else ""

        # ── Path 1: try without forcing structured_output_model ───────────────
        result = self.agent(last_msg)
        self._record_usage(result)
        self.recursion_callback.increment_step()

        parsed = getattr(result, "structured_output", None)
        if parsed is not None:
            logger.info("Context agent: structured output returned natively (fast path 1)")
            return (
                parsed.needs_clarification,
                parsed.context,
                parsed.summary if not parsed.needs_clarification else parsed.clarification_question,
                parsed.clarification_options,
            )

        # ── Path 2: try to parse JSON from the last assistant text ────────────
        from sb3_api.agent.sql_agent import _extract_last_assistant_text  # avoid circular at module level
        last_text = _extract_last_assistant_text(getattr(self.agent, "messages", []))
        if last_text:
            parsed2 = _try_parse_json_from_text(last_text, self._context_model)
            if parsed2 is not None:
                logger.info("Context agent: parsed structured output from assistant text (fast path 2)")
                return (
                    parsed2.needs_clarification,
                    parsed2.context,
                    parsed2.summary if not parsed2.needs_clarification else parsed2.clarification_question,
                    parsed2.clarification_options,
                )

        # ── Path 3: force structured_output_model (fallback) ─────────────────
        logger.warning("Context agent: fast paths failed — falling back to structured_output_model")
        from strands import Agent as StrandsAgent
        fallback_agent = StrandsAgent(
            model=self._strands_model,
            tools=self._strands_tools if hasattr(self, "_strands_tools") else None,
            system_prompt=self.prompt,
            structured_output_model=self._context_model,
            callback_handler=None,
        )
        if len(strands_messages) > 1:
            fallback_agent.messages = strands_messages[:-1]
        result3 = fallback_agent(last_msg, structured_output_model=self._context_model)
        self._record_usage(result3)

        parsed3 = getattr(result3, "structured_output", None)
        if parsed3 is not None:
            return (
                parsed3.needs_clarification,
                parsed3.context,
                parsed3.summary if not parsed3.needs_clarification else parsed3.clarification_question,
                parsed3.clarification_options,
            )

        # Ultimate fallback: treat response as context summary
        text = str(result3)
        return False, text, text[:200], []

    def invoke_agent(
        self,
        messages: Sequence[BaseMessage],
        state: OverallState,
        context: str | None = None,
    ) -> AgentResponse:
        needs_clarification, ctx, final_content, options = self._invoke_and_parse(messages, state)

        state.is_clarification = needs_clarification
        if needs_clarification:
            state.content = final_content
            metadata = {"clarification_options": options}
            trace_type = TraceType.CLARIFICATION
        else:
            state.context = ctx
            metadata = {}
            trace_type = TraceType.CONTEXT

        return AgentResponse(
            traces=[Trace(content=final_content, metadata=metadata, actions=[], type=trace_type)],
            messages=list(messages),
            is_clarification=needs_clarification,
        )

    async def invoke_agent_stream(  # type: ignore[override]
        self,
        messages: Sequence[BaseMessage],
        state: OverallState,
        context: str | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[AgentStreamResponse, None]:
        needs_clarification, ctx, final_content, options = self._invoke_and_parse(messages, state)

        state.is_clarification = needs_clarification
        if needs_clarification:
            state.content = final_content
            metadata = {"clarification_options": options}
            trace_type = TraceType.CLARIFICATION
        else:
            state.context = ctx
            metadata = {}
            trace_type = TraceType.CONTEXT

        if not self.debug_mode:
            yield AgentStreamResponse(
                trace=Trace(content=final_content, metadata=metadata, actions=[], type=trace_type),
                session_id=session_id,
            )

    def get_tools(self) -> list:
        return [
            self.tool_factory.get_knowledge_base_tool(),
            self.tool_factory.get_partial_results_tool(),
        ]

    def get_prompt(self) -> str:
        return self.prompt_registry.get_prompt(PromptId.CONTEXT_AGENT_PROMPT)

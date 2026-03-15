import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from strands import Agent as StrandsAgent
from strands import tool as strands_tool
from strands.models.bedrock import BedrockModel as StrandsBedrockModel

from sb3_api.agent.callbacks.llm_tracker import LLMTracker
from sb3_api.agent.callbacks.recursion_tracker import RecursionTracker
from sb3_api.agent.llm import create_strands_model
from sb3_api.agent.prompts.prompts import PromptRegistry
from sb3_api.agent.tools.factory import ToolFactory
from sb3_api.config.llm import BaseLLMConfig
from sb3_api.models.overall import OverallState
from sb3_api.models.persona import Persona
from sb3_api.models.response import AgentResponse, AgentStreamResponse, Trace

logger = logging.getLogger(__name__)


def _inline_refs(schema: dict, defs: dict) -> dict:
    """Recursively inline all $ref references from $defs into the schema.

    This is required because Strands does not support $ref/$defs in tool
    input schemas. Pydantic's model_json_schema() emits $defs for any
    Literal, Enum, or nested model type.
    """
    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path.startswith("#/$defs/"):
            def_name = ref_path[len("#/$defs/"):]
            if def_name in defs:
                # Inline the definition, then recurse into it
                inlined = dict(defs[def_name])
                inlined.pop("title", None)
                return _inline_refs(inlined, defs)
        return schema  # Unknown $ref – leave as-is

    result: dict = {}
    for key, value in schema.items():
        if key == "$defs":
            continue  # Drop the definitions block after inlining
        elif isinstance(value, dict):
            result[key] = _inline_refs(value, defs)
        elif isinstance(value, list):
            result[key] = [
                _inline_refs(item, defs) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _build_strands_input_schema(lc_tool: Any) -> dict:
    """Convert a LangChain tool's args_schema into a Strands-compatible inputSchema.

    Returns a dict of the form {"json": {...json-schema...}}.
    """
    if lc_tool.args_schema is None:
        # Tools with no schema (e.g. ListSQLDatabaseTool) accept an empty string
        return {
            "json": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }

    raw: dict = lc_tool.args_schema.model_json_schema()
    defs: dict = raw.get("$defs", {})

    # Inline all $refs so Strands sees a flat schema
    inlined = _inline_refs(raw, defs)

    # Clean up fields that confuse Strands
    inlined.pop("title", None)
    inlined.pop("$defs", None)

    # Ensure top-level type is present
    if "type" not in inlined:
        inlined["type"] = "object"

    return {"json": inlined}


def _wrap_langchain_tool(lc_tool: Any) -> Any:
    """Wrap a LangChain BaseTool as a Strands @tool-decorated function.

    Uses lc_tool.invoke() so LangChain handles its own input validation/routing.
    """
    input_schema = _build_strands_input_schema(lc_tool)
    _lc = lc_tool

    def _wrapper(**kwargs: Any) -> dict:
        try:
            # Use the tool's invoke() which handles schema validation internally.
            # For tools with args_schema pass a dict; for bare-string tools pass "".
            if _lc.args_schema is not None:
                result = _lc.invoke(input=kwargs)
            else:
                result = _lc.invoke(input="")
            return {"status": "success", "content": [{"text": str(result)}]}
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", _lc.name)
            return {"status": "error", "content": [{"text": str(exc)}]}

    # Strands requires valid Python identifiers for tool names
    tool_name = lc_tool.name.replace("-", "_")
    return strands_tool(
        name=tool_name,
        description=lc_tool.description,
        inputSchema=input_schema,
    )(_wrapper)


def _messages_to_strands(messages: Sequence[BaseMessage]) -> list[dict]:
    """Convert LangChain BaseMessage list to Strands conversation format."""
    strands_messages: list[dict] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            strands_messages.append(
                {"role": "user", "content": [{"text": str(msg.content)}]}
            )
        elif isinstance(msg, AIMessage):
            content = str(msg.content) if msg.content else "."
            strands_messages.append(
                {"role": "assistant", "content": [{"text": content}]}
            )
        # ToolMessage and SystemMessage are intentionally skipped –
        # they are internal LangChain constructs not needed by Strands.
    return strands_messages


class BaseAgentExecutor(ABC):
    def __init__(  # noqa: PLR0913
        self,
        tool_factory: ToolFactory,
        prompt_registry: PromptRegistry,
        persona: Persona,
        model_config: BaseLLMConfig,
        middleware: list | None = None,
        response_format: Any | None = None,
        context_schema: Any | None = None,
        llm_tracker: LLMTracker | None = None,
    ) -> None:
        self.model_config = model_config
        self.context_schema = context_schema
        self.response_format = response_format
        self.tool_factory = tool_factory
        self.prompt_registry = prompt_registry
        self.middleware = middleware or []
        self.persona = persona
        self.recursion_callback = RecursionTracker()
        self.llm_tracker = llm_tracker if llm_tracker is not None else LLMTracker()

        self.tools = self.get_tools()
        self.prompt = self.get_prompt()
        self.agent = self._create_agent()

    def _format_prompt(self, context: str | None = None) -> str:
        """Return the system prompt, optionally injecting context."""
        prompt = self.get_prompt()
        if context is not None:
            try:
                prompt = prompt.format(context=context)
            except KeyError:
                pass
        return prompt

    def _create_agent(self) -> StrandsAgent:
        """Create a Strands Agent with all LangChain tools wrapped."""
        strands_tools = [_wrap_langchain_tool(t) for t in self.tools]
        return StrandsAgent(
            model=create_strands_model(self.model_config),
            tools=strands_tools if strands_tools else None,
            system_prompt=self.prompt,
            structured_output_model=self.response_format,
            callback_handler=None,
        )

    def _record_usage(self, result: Any) -> None:
        """Record token usage from a Strands AgentResult into LLMTracker."""
        try:
            if self.llm_tracker and hasattr(result, "metrics") and result.metrics:
                usage = result.metrics.accumulated_usage
                self.llm_tracker.record_usage(
                    input_tokens=getattr(usage, "inputTokens", 0),
                    output_tokens=getattr(usage, "outputTokens", 0),
                )
        except Exception:
            logger.debug("Could not record LLM usage metrics", exc_info=True)

    @abstractmethod
    def get_tools(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def create_trace(self, message: BaseMessage, state: OverallState) -> Trace | None:
        raise NotImplementedError

    @staticmethod
    def create_debug_trace(message: BaseMessage) -> Trace | None:
        if isinstance(message, AIMessage):
            return Trace(content=message.text, actions=message.tool_calls, type=message.type)
        if isinstance(message, ToolMessage):
            return Trace(content=message.text, metadata={"name": message.name}, type=message.type)
        return None

    @abstractmethod
    def invoke_agent(
        self,
        messages: Sequence[BaseMessage],
        state: OverallState,
        context: str | None = None,
    ) -> AgentResponse:
        raise NotImplementedError

    @abstractmethod
    async def invoke_agent_stream(
        self,
        messages: Sequence[BaseMessage],
        state: OverallState,
        context: str | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[AgentStreamResponse, None]:
        raise NotImplementedError
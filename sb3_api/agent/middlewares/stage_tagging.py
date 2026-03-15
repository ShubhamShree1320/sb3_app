from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from sb3_api.enums.trace import TraceType


class MessageStageTagMiddleware(AgentMiddleware):
    """Middleware that adds a stage tag to agent and tool messages.

    This middleware propagates "tags" into ToolMessage and AIMessage
    objects so downstream components can identify the message stage (for example,
    TraceType.REASONING or TraceType.INSIGHT).

    Behavior
    - wrap_tool_call: if the invoked tool exposes a "tags" attribute coming from the definition, attach the
      first tag to the produced ToolMessage.
    - wrap_model_call: when the first item of the ModelResponse result is an
      AIMessage, set additional_kwargs["tags"] based on the structure of the model call.

    Notes
    -----
    - The middleware mutates message.additional_kwargs but does not change the
      handlers' return types.

    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tag = getattr(request.tool, "tags", None)
        result = handler(request)
        if isinstance(result, ToolMessage) and tag:
            result.additional_kwargs = {"tags": tag[0]}
        return result

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        if not response.result or not isinstance(response.result[0], AIMessage):
            return response
        message = response.result[0]
        message.additional_kwargs["tags"] = self._determine_tag(message)
        return response

    def _determine_tag(self, message: AIMessage) -> str | None:
        has_content = bool(message.content and message.text.strip())
        has_tool_calls = bool(message.tool_calls)

        if not has_content and has_tool_calls:
            return None

        if has_tool_calls:
            return TraceType.REASONING

        if has_content:
            return TraceType.INSIGHT

        return None

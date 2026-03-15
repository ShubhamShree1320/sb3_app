import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

if TYPE_CHECKING:
    from langchain.agents.structured_output import AutoStrategy, ProviderStrategy, ToolStrategy

logger = logging.getLogger(__name__)


class ResponseFormatMiddleware(AgentMiddleware):
    def __init__(self) -> None:
        self._original_response_format: ToolStrategy | ProviderStrategy | AutoStrategy | None = None
        self._last_request: ModelRequest | None = None
        self._handler: Callable[[ModelRequest], ModelResponse] | None = None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        if request.response_format is not None:
            self._original_response_format = request.response_format

        self._last_request = request
        self._handler = handler

        modified_request = ModelRequest(
            model=request.model,
            system_prompt=request.system_prompt,
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=None,
            state=request.state,
            runtime=request.runtime,
            model_settings=request.model_settings,
        )

        return handler(modified_request)

    def after_agent(self, state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        if self._original_response_format is None or self._handler is None or not runtime.context:
            return None

        if self._last_request:
            final_request = ModelRequest(
                model=self._last_request.model,
                system_prompt=self._last_request.system_prompt,
                messages=state["messages"],
                tools=[],
                tool_choice=None,
                response_format=self._original_response_format,
                state=state,
                runtime=runtime,
                model_settings=self._last_request.model_settings,
            )

        response = self._handler(final_request)

        if hasattr(response, "structured_response"):
            return {"structured_response": response.structured_response}

        return None

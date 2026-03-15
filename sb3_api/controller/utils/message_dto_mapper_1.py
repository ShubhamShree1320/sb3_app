import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from sb3_api.enums.trace import TraceType
from sb3_api.models.message_dto import (
    MessageClarificationDTO,
    MessageContextDTO,
    MessageDTO,
    MessageErrorDTO,
    MessageEvent,
    MessageGenericDTO,
    MessageInsightDTO,
    MessagePlotDTO,
    MessageReasoningDTO,
    MessageSQLDTO,
    MessageSummaryDTO,
    MessageTextDTO,
    MessageUserDTO,
    TraceDTO,
)
from sb3_api.models.overall import OverallState
from sb3_api.models.response import AgentStreamResponse, Trace
from sb3_api.models.session import SessionChat
from sb3_api.models.session_dto import SessionChatDTO

logger = logging.getLogger(__name__)


def _create_generic_dto(trace: Trace) -> MessageDTO:
    return MessageGenericDTO(content=trace.model_dump_json())


def _map_tool_to_dto(trace: Trace) -> MessageDTO:
    """Map a Trace object to an appropriate MessageDTO based on the tool type.

    Raises:
        json.JSONDecodeError: If the trace content cannot be parsed as valid JSON (plots).

    """
    if trace.metadata.get("name") == "generate_plot":
        return _map_plot_trace_to_dto(trace)
    return _create_generic_dto(trace)


def _map_plot_trace_to_dto(trace: Trace) -> MessageDTO:
    try:
        data = json.loads(trace.content)
        return MessagePlotDTO(data=data)
    except json.JSONDecodeError:
        logger.exception("Error mapping plot message")
        raise


_TRACE_TYPE_HANDLERS: dict[str, Callable[[Trace], MessageDTO]] = {
    TraceType.HUMAN: lambda t: MessageUserDTO(content=t.content),
    TraceType.AI: lambda t: MessageTextDTO(content=t.content)
    if t.content
    else _create_generic_dto(t),  # for debugging profile
    TraceType.TOOL: _map_tool_to_dto,  # for debugging profile
    TraceType.CONTEXT: lambda t: MessageContextDTO(content=t.content),
    TraceType.CLARIFICATION: lambda t: MessageClarificationDTO(
        content=t.content, options=t.metadata.get("clarification_options", [])
    ),
    TraceType.REASONING: lambda t: MessageReasoningDTO(content=t.content),
    TraceType.SQL: lambda t: MessageSQLDTO(content=t.content),
    TraceType.PLOT: _map_plot_trace_to_dto,
    TraceType.INSIGHT: lambda t: MessageInsightDTO(content=t.content),
    TraceType.ERROR: lambda t: MessageErrorDTO(content=t.content),
}

_BASE_MESSAGE_HANDLERS: dict[type[BaseMessage], Callable[[BaseMessage], MessageDTO]] = {
    HumanMessage: lambda m: MessageUserDTO(content=m.text),
    AIMessage: lambda m: MessageTextDTO(content=m.text),
}


def map_trace_to_message_dto(trace: Trace) -> MessageDTO:
    handler = _TRACE_TYPE_HANDLERS.get(trace.type)
    return _create_generic_dto(trace) if handler is None else handler(trace)


def map_agent_response_to_message_event(
    session_id: str, agent_response: AgentStreamResponse | dict[str, Any] | Any
) -> MessageEvent:
    """Map an agent response to a MessageEvent.

    Accepts either a typed AgentStreamResponse or an untyped dict/object,
    and casts it to AgentStreamResponse for type safety.
    """
    if not isinstance(agent_response, AgentStreamResponse):
        agent_response = AgentStreamResponse.model_validate(agent_response)

    return MessageEvent(session_id=session_id, message=map_trace_to_message_dto(agent_response.trace))


def map_base_message_to_dto(message: BaseMessage) -> MessageDTO:
    handler = _BASE_MESSAGE_HANDLERS.get(type(message))
    return MessageTextDTO(content=message.text) if handler is None else handler(message)


def map_session_chat_to_dto(session_chat: SessionChat) -> SessionChatDTO:
    return SessionChatDTO(
        session_id=session_chat.session_id,
        messages=[map_base_message_to_dto(msg) for msg in session_chat.content],
    )


def map_overall_state_to_message_dto(overall_state: OverallState) -> MessageSummaryDTO:
    plot = None
    if isinstance(overall_state.plot, str):
        try:
            plot = json.loads(overall_state.plot)
        except json.JSONDecodeError:
            logger.exception("Failed to parse plot JSON")

    traces = overall_state.traces
    human_query = overall_state.messages[0].text if overall_state.messages else None
    return MessageSummaryDTO(
        content=overall_state.content,
        human_query=human_query,
        sql_query=overall_state.sql_query,
        reasoning=overall_state.reasoning_summary,
        session_id=overall_state.session_id,
        plot=plot,
        is_clarification=overall_state.is_clarification,
        traces=[TraceDTO(type=trace.type, content=trace.content) for trace in traces],
    )


def map_graph_state_to_message_event(session_id: str, graph_state: dict) -> MessageEvent:
    overall_state = OverallState.model_validate(graph_state)
    return MessageEvent(session_id=session_id, message=map_overall_state_to_message_dto(overall_state))

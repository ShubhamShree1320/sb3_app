from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MessageTextDTO(BaseModel):
    """Universal text message from AI. Can be plain text or markdown."""

    type: Literal["text"] = "text"
    content: str


class MessageUserDTO(BaseModel):
    """Message originating from user input."""

    type: Literal["user"] = "user"
    content: str


class MessageContextDTO(BaseModel):
    """Information about the context."""

    type: Literal["context"] = "context"
    content: str


class MessageClarificationDTO(BaseModel):
    """Information about the clarification."""

    type: Literal["clarification"] = "clarification"
    content: str
    options: list[str] = Field(default_factory=list, description="List of clarification options")


class MessageReasoningDTO(BaseModel):
    """Information about the reasoning behind the response."""

    type: Literal["reasoning"] = "reasoning"
    content: str


class MessageSQLDTO(BaseModel):
    """Information about the SQL query related to the response."""

    type: Literal["sql"] = "sql"
    content: str


class MessagePlotDTO(BaseModel):
    """Message that carries data in the format of a chart.js ChartConfiguration."""

    type: Literal["plot"] = "plot"
    data: dict = Field(default_factory=dict)


class MessageInsightDTO(BaseModel):
    """The last agent message containing insights."""

    type: Literal["insight"] = "insight"
    content: str


class MessageErrorDTO(BaseModel):
    """Universal error message."""

    type: Literal["error"] = "error"
    content: str


class TraceDTO(BaseModel):
    """Trace information."""

    type: str
    content: str


class MessageSummaryDTO(BaseModel):
    """Summary of the response."""

    type: Literal["summary"] = "summary"
    content: str | None
    human_query: str | None
    sql_query: str | None
    reasoning: str | None
    session_id: str | None
    plot: dict | None = None
    is_clarification: bool | None
    traces: list[TraceDTO]


class MessageGenericDTO(BaseModel):
    """Message with unspecified text content."""

    type: Literal["generic"] = "generic"
    content: str


# Union type with discriminator annotation
MessageDTO = Annotated[
    MessageUserDTO
    | MessageTextDTO
    | MessageContextDTO
    | MessageClarificationDTO
    | MessageReasoningDTO
    | MessageSQLDTO
    | MessagePlotDTO
    | MessageInsightDTO
    | MessageErrorDTO
    | MessageSummaryDTO
    | MessageGenericDTO,
    Field(discriminator="type"),
]


# SSE payload envelope
class MessageEvent(BaseModel):
    session_id: str
    message: MessageDTO


# Non-streaming envelope
class MessageResponse(BaseModel):
    session_id: str
    messages: list[MessageDTO]

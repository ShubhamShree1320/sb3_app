from operator import add
from typing import Annotated

from pydantic import Field

from sb3_api.models.input import InputState
from sb3_api.models.response import Trace


class OverallState(InputState):
    traces: Annotated[list[Trace], add] = Field(default_factory=list)
    context: str | None = None
    content: str | None = None
    plot: str | None = None
    summary: str | None = None
    sql_query: str | None = None
    reasoning_summary: str | None = None
    partial_response: str | None = None
    is_clarification: bool = False

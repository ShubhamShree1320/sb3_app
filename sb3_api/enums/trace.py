from enum import Enum


class TraceType(str, Enum):
    """Trace types for agent responses."""

    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"

    CONTEXT = "sources"
    CLARIFICATION = "clarification"
    REASONING = "reasoning"
    SQL = "sql"
    PLOT = "plot"
    INSIGHT = "insight"
    ERROR = "error"

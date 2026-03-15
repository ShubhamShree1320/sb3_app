from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from sb3_api.enums.trace import TraceType


class Persona(str, Enum):
    BUSINESS = "business"
    ANALYST = "analyst"


class PersonaStage(BaseModel):
    model_config = ConfigDict(frozen=True)

    business_stages: frozenset[TraceType] = Field(
        default=frozenset({TraceType.CONTEXT, TraceType.PLOT, TraceType.INSIGHT, TraceType.REASONING}),
        description="Stages visible to business users",
    )
    analyst_stages: frozenset[TraceType] = Field(
        default=frozenset({TraceType.CONTEXT, TraceType.SQL, TraceType.PLOT, TraceType.INSIGHT, TraceType.REASONING}),
        description="Stages visible to analysts",
    )

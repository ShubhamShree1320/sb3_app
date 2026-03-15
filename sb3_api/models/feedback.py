from datetime import datetime
from typing import Self, cast
from uuid import UUID

from pydantic import BaseModel, Field

from sb3_api.enums.evaluation import Evaluation
from sb3_api.repository.feedback.model import BaseFeedback


class FeedbackBase(BaseModel):
    evaluation: Evaluation = Field(description="Evaluation of the agent's response")


class FeedbackRecord(FeedbackBase):
    query: str = Field(description="Question asked by a user", min_length=1)
    result: str = Field(description="Answer given by the agent (SQL query)", min_length=1)
    session_id: UUID | None = Field(
        default=None, examples=[None], description="Session ID for which the feedback was recorded"
    )


class FeedbackRecordOutput(FeedbackRecord):
    id: UUID
    updated_at: datetime

    @classmethod
    def from_data_model(cls, data: BaseFeedback) -> Self:
        return cls(
            id=cast("UUID", data.id),
            session_id=cast("UUID", data.session_id),
            evaluation=cast("Evaluation", data.evaluation),
            query=cast("str", data.query),
            result=cast("str", data.result),
            updated_at=cast("datetime", data.updated_at),
        )

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


class SessionChat(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str  # partition key
    user: str  # GSI partition key
    created_at: str = Field(serialization_alias="createdAt", validation_alias="createdAt")
    content: list[BaseMessage] = Field(default_factory=list, min_length=1)

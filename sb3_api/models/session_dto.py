from pydantic import BaseModel, Field

from sb3_api.models.message_dto import MessageDTO


class SessionInfoDTO(BaseModel):
    session_id: str
    query: str
    timestamp: str

    @classmethod
    def from_dynamo_item(cls, item: dict) -> "SessionInfoDTO":
        return cls(
            session_id=item["session_id"], query=item["content"][0]["data"]["content"], timestamp=item["createdAt"]
        )


class SessionChatDTO(BaseModel):
    session_id: str
    messages: list[MessageDTO] = Field(default_factory=list, min_length=1)

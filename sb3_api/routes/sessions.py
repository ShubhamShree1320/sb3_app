from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path

from sb3_api.auth.auth import require_auth
from sb3_api.controller.utils import message_dto_mapper
from sb3_api.dependencies import get_session_repository
from sb3_api.models.response import UserInfo
from sb3_api.models.session_dto import SessionChatDTO, SessionInfoDTO
from sb3_api.repository.session.dynamodb import DynamoDBSessionRepository

router = APIRouter()


@router.get(
    "/session/{session_id}",
    summary="Get a chat session",
)
async def get_session(
    session_id: Annotated[UUID, Path(description="Session ID")],
    session_repository: Annotated[DynamoDBSessionRepository, Depends(get_session_repository)],
    user: Annotated[UserInfo, Depends(require_auth)],
) -> SessionChatDTO:
    session_chat = session_repository.get_session(session_id, user.email)
    return message_dto_mapper.map_session_chat_to_dto(session_chat)


@router.get(
    "/session",
    summary="Get all sessions for a user",
)
async def get_user_sessions(
    session_repository: Annotated[DynamoDBSessionRepository, Depends(get_session_repository)],
    user: Annotated[UserInfo, Depends(require_auth)],
) -> list[SessionInfoDTO]:
    return session_repository.get_sessions_by_user(user.email)

import datetime
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from sb3_api.controller.utils import message_dto_mapper
from sb3_api.exceptions.exceptions import SessionNotFoundError
from sb3_api.models.message_dto import MessageErrorDTO, MessageEvent
from sb3_api.models.persona import Persona
from sb3_api.models.session import SessionChat
from sb3_api.repository.session.dynamodb import DynamoDBSessionRepository

logger = logging.getLogger(__name__)


class ChatController:
    """Controller for handling chat operations."""

    def __init__(self, graph: Any, session_repository: DynamoDBSessionRepository) -> None:
        self.graph = graph
        self.session_repository = session_repository

    def get_or_create_session(
        self,
        session_id: str | None,
        user_email: str,
    ) -> SessionChat:
        """Get existing session or create a new one."""
        if session_id is not None:
            try:
                return self.session_repository.get_session(uuid.UUID(session_id), user_email)
            except SessionNotFoundError:
                logger.exception("Session '%s' not found. Creating new session.", session_id)
        session_id_new = uuid.uuid4()
        return SessionChat(
            session_id=str(session_id_new),
            user=user_email,
            created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        )

    def _create_error_message_event(
        self, session_id: str, message: str = "Unacceptable issue occurred, please try again"
    ) -> MessageEvent:
        return MessageEvent(session_id=session_id, message=MessageErrorDTO(content=message))

    async def process_chat_query_stream(
        self,
        query: str,
        persona: Persona,
        user_email: str,
        session_id: str | None,
        debug_mode: bool = False,  # noqa: FBT001, FBT002
    ) -> AsyncGenerator[MessageEvent, None]:
        """Process a chat query and stream the response."""
        session = self.get_or_create_session(session_id, user_email)
        session.content.append(HumanMessage(content=query))

        final_state = None
        try:
            initial_input = {
                "session_id": session.session_id,
                "messages": session.content,
                "persona": persona,
                "debug_mode": debug_mode,
            }

            async for state_snapshot in self.graph.astream(
                initial_input,
                stream_mode=["custom", "values"],
            ):
                # astream yields (mode, value) tuples
                if isinstance(state_snapshot, tuple):
                    mode, value = state_snapshot
                    if mode == "values":
                        final_state = value
                    elif mode == "custom":
                        try:
                            yield message_dto_mapper.map_agent_response_to_message_event(
                                session_id=session.session_id, agent_response=value
                            )
                        except Exception:
                            logger.exception("Mapping error for session %s", session.session_id)
                            yield self._create_error_message_event(session.session_id)

            if final_state:
                yield message_dto_mapper.map_graph_state_to_message_event(session.session_id, final_state)

                insight = final_state.get("content", "")
                if insight:
                    session.content.append(AIMessage(content=insight))
                    self.session_repository.save_session(session)

        except Exception:
            logger.exception("Streaming error for session %s", session.session_id)
            yield self._create_error_message_event(session.session_id)
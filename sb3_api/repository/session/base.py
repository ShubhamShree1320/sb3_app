import logging
from abc import ABC, abstractmethod
from uuid import UUID

from sb3_api.models.session import SessionChat
from sb3_api.models.session_dto import SessionInfoDTO

logger = logging.getLogger(__name__)


class SessionRepository(ABC):
    """Abstract base class for session storage."""

    @abstractmethod
    def get_session(self, session_id: UUID, user: str) -> SessionChat:
        raise NotImplementedError

    @abstractmethod
    def save_session(self, session: SessionChat) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_sessions_by_user(self, user: str) -> list[SessionInfoDTO]:
        raise NotImplementedError

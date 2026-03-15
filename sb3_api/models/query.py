import datetime
import uuid

from pydantic import BaseModel

from sb3_api.models.persona import Persona
from sb3_api.models.session import SessionChat


class UserQuery(BaseModel):
    query: str
    profile: Persona
    debug_mode: bool = False
    session_id: uuid.UUID | None = None

    def to_session(self, user_email: str) -> SessionChat:
        session_id = self.session_id or uuid.uuid4()
        return SessionChat(
            session_id=str(session_id),
            user=str(user_email),
            created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        )


class ExchangeTokenRequest(BaseModel):
    authorization_code: str
    redirect_uri: str | None = None

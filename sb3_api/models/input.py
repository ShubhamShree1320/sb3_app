from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from sb3_api.models.persona import Persona


class InputState(BaseModel):
    messages: list[BaseMessage]
    session_id: str | None = None
    persona: Persona = Persona.BUSINESS
    debug_mode: bool = False

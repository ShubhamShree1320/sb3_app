import re
from typing import Any

from sb3_api.enums.trace import TraceType
from sb3_api.models.persona import Persona, PersonaStage


def truncate_string(value: Any, length: int, *, suffix: str = "...") -> str:
    if not isinstance(value, str) or length <= 0 or len(value) <= length:
        return value

    return value[: length - len(suffix)].rsplit(" ", 1)[0] + suffix


def should_display(stage: TraceType | str, persona: Persona) -> bool:
    if isinstance(stage, str):
        try:
            stage = TraceType(stage)
        except ValueError:
            return False

    stages = PersonaStage().business_stages if persona == Persona.BUSINESS else PersonaStage().analyst_stages
    return stage in stages


def validate_sql_identifier(identifier: str) -> str:
    """Validate that an identifier (table name, column name) is safe to use in SQL."""
    if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
        msg = f"Invalid SQL identifier: {identifier}."
        raise ValueError(msg)
    return identifier

import json
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sb3_api.auth.auth import require_auth
from sb3_api.controller.chat_controller import ChatController
from sb3_api.dependencies import get_chat_controller
from sb3_api.models.message_dto import MessageEvent, MessageResponse
from sb3_api.models.persona import Persona
from sb3_api.models.query import UserQuery
from sb3_api.models.response import UserInfo

logger = logging.getLogger(__name__)

router = APIRouter()

EXAMPLE_QUERY = "How many TV-Acquisition orders were placed per month in 2024"

sse_schema_description = """
**Server-Sent Events (SSE) Stream**

This endpoint returns a stream of message events.

**Event Format:**

We define the following two event types:

- Data event: `data: <payload>`
  - Prefix **data:** followed by a space character.
  - Payload of **{message_response_class_name}** as stringified JSON. See the schema in this section
    to learn more about the structure.
- Completion marker: `[DONE]`

**How to consume:**
1. Connect using an EventSource or SSE client
2. Parse the payload of each `data:` line as JSON
3. Handle based on the type of the payload
4. Stream ends when `[DONE]` is received

"""


def sse_response_schema(message_response_class: type[BaseModel]) -> dict:
    """Generate OpenAPI response schema that includes DTO schemas for SSE streams of chat messages."""
    # Generate full schema with all nested definitions.
    full_schema = message_response_class.model_json_schema(
        mode="serialization", ref_template="#/components/schemas/{model}"
    )

    # Extract the main schema and definitions separately.
    schema_without_defs = {k: v for k, v in full_schema.items() if k != "$defs"}
    defs = full_schema.get("$defs", {})

    return {
        "responses": {
            200: {
                "description": sse_schema_description.format(
                    message_response_class_name=message_response_class.__name__
                ),
                "content": {"text/event-stream": {"schema": {**schema_without_defs, "$defs": defs}}},
            }
        }
    }


@router.post(
    "/sql-agent/chat",
    summary="Return a response based on a generated sql query from natural language query",
)
async def sql_agent(
    body: Annotated[
        UserQuery,
        Body(examples=[UserQuery(query=EXAMPLE_QUERY, profile=Persona.ANALYST, session_id=None)]),
    ],
    chat_controller: Annotated[ChatController, Depends(get_chat_controller)],
    user: Annotated[UserInfo, Depends(require_auth)],
) -> MessageResponse:
    messages = []

    async for message_event in chat_controller.process_chat_query_stream(
        query=body.query,
        session_id=str(body.session_id) if body.session_id else None,
        user_email=user.email,
        persona=body.profile,
        debug_mode=body.debug_mode,
    ):
        session_id = message_event.session_id
        messages.append(message_event.message)

    return MessageResponse(session_id=session_id, messages=messages)


@router.post(
    "/sql-agent/chat/stream",
    summary="Stream response based on a generated sql query from natural language query using SSE",
    response_class=StreamingResponse,
    **sse_response_schema(MessageEvent),
)
async def sql_agent_stream(
    body: Annotated[
        UserQuery,
        Body(examples=[UserQuery(query=EXAMPLE_QUERY, profile=Persona.ANALYST, session_id=None)]),
    ],
    chat_controller: Annotated[ChatController, Depends(get_chat_controller)],
    user: Annotated[UserInfo, Depends(require_auth)],
) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str]:
        """Generate SSE-formatted strings from event objects."""
        try:
            async for message_event in chat_controller.process_chat_query_stream(
                query=body.query,
                session_id=str(body.session_id) if body.session_id else None,
                user_email=user.email,
                persona=body.profile,
                debug_mode=body.debug_mode,
            ):
                try:
                    yield f"data: {message_event.model_dump_json()}\n\n"
                except AttributeError:
                    # Handle dict objects (e.g., error responses)
                    yield f"data: {json.dumps(message_event, default=str)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

import uuid
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path

from sb3_api.auth.auth import require_auth
from sb3_api.dependencies import get_feedback_store
from sb3_api.models.feedback import FeedbackBase, FeedbackRecord, FeedbackRecordOutput
from sb3_api.stores.feedback_store import FeedbackStore

router = APIRouter()


@router.post(
    "/feedback",
    summary="Provide feedback",
    status_code=HTTPStatus.CREATED,
)
async def give_feedback(
    body: Annotated[FeedbackRecord, Body()],
    records_store: Annotated[FeedbackStore, Depends(get_feedback_store)],
    _: Annotated[None, Depends(require_auth)],
) -> FeedbackRecordOutput:
    """Provide an evaluation of a single response from the agent."""
    if body.session_id is None:
        body.session_id = uuid.uuid4()

    return await records_store.save_feedback(data=body)


@router.put(
    "/feedback/{feedback_id}",
    summary="Update feedback",
)
async def update_feedback(
    feedback_id: Annotated[uuid.UUID, Path(description="Feedback ID")],
    body: Annotated[FeedbackBase, Body()],
    records_store: Annotated[FeedbackStore, Depends(get_feedback_store)],
    _: Annotated[None, Depends(require_auth)],
) -> FeedbackRecordOutput:
    return await records_store.update_feedback(feedback_id=feedback_id, data=body)


@router.get(
    "/feedback/{feedback_id}",
    summary="Get feedback",
)
async def get_feedback(
    feedback_id: Annotated[uuid.UUID, Path(description="Feedback ID")],
    records_store: Annotated[FeedbackStore, Depends(get_feedback_store)],
    _: Annotated[None, Depends(require_auth)],
) -> FeedbackRecordOutput:
    return await records_store.get_feedback(feedback_id=feedback_id)


@router.get(
    "/feedback/{session_id}/",
    summary="Get feedbacks for a session",
)
async def get_session_feedback(
    session_id: Annotated[uuid.UUID, Path(description="Feedback ID")],
    records_store: Annotated[FeedbackStore, Depends(get_feedback_store)],
    _: Annotated[None, Depends(require_auth)],
) -> list[FeedbackRecordOutput]:
    return await records_store.get_session_feedback(session_id=session_id)


@router.delete(
    "/feedback/{feedback_id}",
    summary="Delete feedback record",
)
async def delete_feedback(
    feedback_id: Annotated[uuid.UUID, Path(description="Feedback ID")],
    records_store: Annotated[FeedbackStore, Depends(get_feedback_store)],
    _: Annotated[None, Depends(require_auth)],
) -> None:
    return await records_store.delete_feedback(feedback_id=feedback_id)

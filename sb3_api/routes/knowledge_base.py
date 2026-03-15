from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from sb3_api.agent.tools.knowledge_base.knowledge_base import KnowledgeBase
from sb3_api.auth.auth import require_admin_auth
from sb3_api.dependencies import get_knowledge_base
from sb3_api.enums.document import DocumentType
from sb3_api.exceptions.exceptions import CollectionNotFoundError
from sb3_api.models.knowledge_base import CollectionResponse

router = APIRouter()


@router.post("/collections/{doc_type}/create", summary="Create or load a collection", status_code=HTTPStatus.CREATED)
def create_collection(
    doc_type: DocumentType,
    kb: Annotated[KnowledgeBase, Depends(get_knowledge_base)],
    _: Annotated[None, Depends(require_admin_auth)],
) -> CollectionResponse:
    existed_before = kb.ensure_collection(doc_type)
    kb.create_or_load_collection(doc_type)
    if existed_before:
        return CollectionResponse(collection=doc_type, action="loaded")
    return CollectionResponse(collection=doc_type, action="created")


@router.post("/collections/{doc_type}/recreate", summary="Recreate a collection", status_code=HTTPStatus.CREATED)
def recreate_collection(
    doc_type: DocumentType,
    kb: Annotated[KnowledgeBase, Depends(get_knowledge_base)],
    _: Annotated[None, Depends(require_admin_auth)],
) -> CollectionResponse:
    kb.delete_collection(doc_type)
    kb.create_or_load_collection(doc_type)
    return CollectionResponse(collection=doc_type, action="recreated")


@router.delete("/collections/{doc_type}", summary="Delete a collection", status_code=HTTPStatus.OK)
def delete_collection(
    doc_type: DocumentType,
    kb: Annotated[KnowledgeBase, Depends(get_knowledge_base)],
    _: Annotated[None, Depends(require_admin_auth)],
) -> CollectionResponse:
    try:
        kb.delete_collection(doc_type)
        return CollectionResponse(collection=doc_type, action="deleted")
    except CollectionNotFoundError as e:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Collection {doc_type.value} not found",
        ) from e

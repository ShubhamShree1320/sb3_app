from fastapi import APIRouter

from sb3_api.routes.auth import router as auth_router
from sb3_api.routes.chat import router as sql_agent_router
from sb3_api.routes.feedback import router as feedback_router
from sb3_api.routes.knowledge_base import router as knowledge_base_router
from sb3_api.routes.sessions import router as session_router


def create_router(prefix: str) -> APIRouter:
    router = APIRouter(prefix=prefix)

    router.include_router(sql_agent_router, tags=["SQL agent"])
    router.include_router(session_router, tags=["Session Store"])
    router.include_router(feedback_router, tags=["feedback"])
    router.include_router(knowledge_base_router, tags=["knowledge base"])
    router.include_router(auth_router, tags=["auth"])

    return router

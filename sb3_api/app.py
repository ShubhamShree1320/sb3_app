from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from sb3_api.config.logging import setup_logging
from sb3_api.dependencies import get_feedback_repository, get_knowledge_base, get_settings
from sb3_api.exceptions.exception_handlers import (
    generic_exception_handler,
    not_found_exception_handler,
)
from sb3_api.exceptions.exceptions import NotFoundError
from sb3_api.router import create_router
from sb3_api.settings import ServiceSettings
from sb3_api.dependencies import get_sql_database
from sb3_api.agent.tools.factory import ToolFactory

setup_logging()
# Add to your app startup (e.g., main.py or app.py)
import logging
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def get_lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    # Create table (will not attempt to recreate tables already present)
    repository = get_feedback_repository(settings=get_settings())
    await repository.create_schema()

    # Create and populate knowledge base collections if quota not available comment out for now
    # knowledge_base = get_knowledge_base(settings=get_settings())
    # #knowledge_base.create_all_collections()
    # if knowledge_base is not None:
    #     knowledge_base.create_all_collections()

    yield

def create_app(
    settings: ServiceSettings = get_settings(),
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager] | None = None,
) -> FastAPI:
    # Initialize FastAPI app
    app = FastAPI(
        title=settings.SERVICE_NAME,
        description="API for an agent interacting with Amazon Redshift data",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Add the frontend URL
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Redirect root path to documentation
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    app.include_router(create_router(prefix=settings.API_PREFIX))

    app.add_exception_handler(Exception, generic_exception_handler)
    app.add_exception_handler(NotFoundError, not_found_exception_handler)  # type: ignore[arg-type]

    return app

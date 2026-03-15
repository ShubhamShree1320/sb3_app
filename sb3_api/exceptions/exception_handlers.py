import logging
from http import HTTPStatus
from typing import Self

from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sb3_api.exceptions.exceptions import NotFoundError

log = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    detail: str

    @classmethod
    def from_exception(cls, exc: Exception) -> Self:
        return cls(detail=str(exc))


async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    log.warning("Caught %s due to: %s", exc.__class__.__name__, exc)
    error = ErrorResponse(detail="Internal Server Error")
    return JSONResponse(content=error.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


async def not_found_exception_handler(_: Request, exc: NotFoundError) -> JSONResponse:
    log.warning("Caught %s due to: %s", exc.__class__.__name__, exc)
    error = ErrorResponse.from_exception(exc)
    return JSONResponse(content=error.model_dump(), status_code=HTTPStatus.NOT_FOUND)

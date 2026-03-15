import logging
import urllib.parse
from http import HTTPStatus
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from sb3_api.auth.auth import get_user_info, require_auth
from sb3_api.dependencies import ServiceSettings, get_settings
from sb3_api.models.query import ExchangeTokenRequest
from sb3_api.models.response import (
    AuthorizeResponse,
    CheckTokenResponse,
    ExchangeTokenResponse,
    UserInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth")


@router.get("/authorize", response_model=AuthorizeResponse)
async def authorize(
    settings: Annotated[ServiceSettings, Depends(get_settings)],
    redirect_uri: Annotated[str | None, Query(description="OAuth2 redirect URI")] = None,
    state: Annotated[str | None, Query(description="OAuth2 state parameter for CSRF protection")] = None,
) -> dict:
    """OAuth2 Authorization Endpoint.

    Initiates the OAuth2 authorization code flow by providing the authorization URL
    that the client should redirect the user to.
    """
    # Use redirect_uri from settings if not provided
    actual_redirect_uri = redirect_uri or settings.AUTH_REDIRECT_URI

    params = {
        "response_type": "code",
        "client_id": settings.AUTH_CLIENT_ID,
        "redirect_uri": actual_redirect_uri,
        "state": state or "default_state",
    }

    authorization_url = f"{settings.AUTH_AUTHORIZATION_URL}?{urllib.parse.urlencode(params)}"

    return {
        "authorization_url": authorization_url,
        "state": params["state"],
        "client_id": settings.AUTH_CLIENT_ID,
        "redirect_uri": actual_redirect_uri,
    }


@router.post("/check-token", response_model=CheckTokenResponse)
async def check_token(user_info: Annotated[UserInfo, Depends(require_auth)]) -> dict:
    return {"valid": True, "user_info": user_info}


@router.post("/exchange-token")
async def exchange_token(
    request: ExchangeTokenRequest, settings: Annotated[ServiceSettings, Depends(get_settings)]
) -> ExchangeTokenResponse:
    """Exchange authorization code for access token."""
    # Use redirect_uri from settings if not provided
    redirect_uri = request.redirect_uri or settings.AUTH_REDIRECT_URI

    data = {
        "client_id": settings.AUTH_CLIENT_ID,
        "client_secret": settings.AUTH_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": request.authorization_code,
        "redirect_uri": redirect_uri,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.AUTH_TOKEN_URL,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            token_response = response.json()
            user_info = await get_user_info(settings, token_response["access_token"])

        return ExchangeTokenResponse(
            access_token=token_response["access_token"],
            user_info=user_info,
            expires_in=token_response["expires_in"],
        )

    except httpx.HTTPStatusError as e:
        msg = f"HTTP error during token exchange: {e.response.status_code} - {e.response.text}"
        logger.exception(msg)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e
    except HTTPException:
        # Re-raise HTTPException
        raise
    except Exception as e:
        logger.exception("Unexpected error during token exchange")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)) from e

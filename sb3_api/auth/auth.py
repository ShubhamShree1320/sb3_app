import logging
from http import HTTPStatus

import httpx
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from sb3_api.dependencies import ServiceSettings, get_settings
from sb3_api.models.response import UserInfo

logger = logging.getLogger(__name__)

bearer_token = HTTPBearer(auto_error=False)  # Don't auto-error to handle auth disabled case.


def validate_jwt_audience(token: str, settings: ServiceSettings) -> None:
    """Check the audience claim of the provided JWT token.

    Raises:
        HTTPException: If token is invalid or audience doesn't match

    """
    if not settings.AUTH_AUDIENCE:
        # If no audience is configured, skip JWT validation.
        logger.warning("AUTH_AUDIENCE not configured - skipping JWT audience validation")
        return

    try:
        decoded_token = jwt.decode(
            token,
            options={"verify_signature": False},  # Token is verified by Corproot Identity Service.
            audience=settings.AUTH_AUDIENCE,
            algorithms=["RS256", "HS256"],  # Common OAuth2 algorithms.
        )

        if "aud" in decoded_token:
            audiences = decoded_token["aud"]
            if isinstance(audiences, str):
                audiences = [audiences]

            if settings.AUTH_AUDIENCE not in audiences:
                msg = f"Token audience {audiences} does not match expected {settings.AUTH_AUDIENCE}"
                logger.error(msg)
                raise_auth_error(f"Invalid audience claim. Expected: {settings.AUTH_AUDIENCE}")
        else:
            logger.error("Token missing audience claim")
            raise_auth_error("Token missing audience claim")

        msg = f"JWT token validated successfully for audience: {settings.AUTH_AUDIENCE}"
        logger.info(msg)

    except jwt.InvalidAudienceError:
        logger.exception("Invalid audience in token")
        raise_auth_error("Invalid audience claim")
    except jwt.InvalidTokenError as e:
        logger.exception("Invalid JWT token")
        raise_auth_error(f"Invalid token: {e!s}")
    except HTTPException:
        # Re-raise HTTPException without wrapping
        raise
    except Exception:
        logger.exception("Unexpected error validating JWT token")
        raise_auth_error("Token validation failed")


def is_user_allowed(user_email: str, settings: ServiceSettings) -> bool:
    allowed_emails = (
        [email.strip().lower() for email in settings.AUTH_ALLOWED_USERS.split(",")]
        if settings.AUTH_ALLOWED_USERS
        else []
    )
    return user_email.lower() in allowed_emails


async def get_user_info(settings: ServiceSettings, token: str | None = None) -> UserInfo | None:
    if not token:
        return None

    validate_jwt_audience(token, settings)

    async with httpx.AsyncClient() as client:
        response = await client.get(settings.AUTH_USER_INFO_URL, headers={"Authorization": f"Bearer {token}"})
        if response.status_code == HTTPStatus.OK:
            user_info = response.json()
            if not is_user_allowed(user_info["email"], settings):
                msg = f"Access denied for {user_info['email']} - not in allowed users list"
                logger.warning(msg)
                raise HTTPException(
                    status_code=HTTPStatus.FORBIDDEN,
                    detail="Access forbidden: user not in allowed users list",
                )
            return UserInfo(
                user_name=user_info["user_name"],
                given_name=user_info["given_name"],
                family_name=user_info["family_name"],
                email=user_info["email"],
                department=user_info["user_attributes"]["department"][0],
            )
        if response.status_code == HTTPStatus.UNAUTHORIZED:
            return None
        response.raise_for_status()
        return None


async def require_auth(
    settings: ServiceSettings = Depends(get_settings),
    token: HTTPAuthorizationCredentials | None = Depends(bearer_token),
) -> UserInfo:
    """Dependency to get current user from Authorization header.
    This will raise HTTPException if token is invalid or missing.
    This is for protected routes that require authentication.
    """
    # If authentication is disabled, return a dummy user.
    if settings.AUTH_DISABLED:
        logger.info("Authentication disabled - allowing access without authentication")
        return UserInfo(
            user_name="anonymous",
            given_name="Anonymous",
            family_name="User",
            email="anonymous@localhost",
            department="N/A",
        )

    # If auth is enabled, token is required.
    if not token:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_info = await get_user_info(settings, token.credentials)
    if user_info is None:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    msg = f"Access by {user_info.email}"
    logger.info(msg)
    return user_info


async def require_admin_auth(
    user: UserInfo = Depends(require_auth),
    settings: ServiceSettings = Depends(get_settings),
) -> UserInfo:
    """Dependency to get current user from Authorization header.
    This will raise HTTPException if token is invalid or missing.
    This is for protected routes that require admin authentication.
    """
    allowed_admins = (
        [email.strip().lower() for email in settings.AUTH_ADMIN_USERS.split(",")] if settings.AUTH_ADMIN_USERS else []
    )

    if user.email.lower() not in allowed_admins:
        msg = f"Admin access denied for {user.email}"
        logger.warning(msg)
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Access forbidden: admin privileges required",
        )

    return user


def raise_auth_error(detail: str) -> None:
    raise HTTPException(
        status_code=HTTPStatus.UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )

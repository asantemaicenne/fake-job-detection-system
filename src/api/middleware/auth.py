import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from configs.settings import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")


class Token(BaseModel):
    """Schema returned upon successful authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    """Decoded JWT payload containing identity and authority claims."""
    sub: str = Field(..., description="Username or subject identifier")
    role: str = Field(default="api_user", description="Assigned RBAC role")
    exp: Optional[int] = None


class User(BaseModel):
    """Runtime representation of the authenticated user."""
    username: str
    role: str
    is_active: bool = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a stored bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generates a secure bcrypt hash for credential storage."""
    return pwd_context.hash(password)


def create_access_token(
    subject: str,
    role: str = "api_user",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Encodes identity claims and expiry into a cryptographically signed JWT.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Extracts, decodes, and validates the bearer token from incoming headers.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValueError):
        raise credentials_exception

    return User(
        username=token_data.sub,
        role=token_data.role,
        is_active=True,
    )


def require_role(allowed_roles: List[str]) -> Callable:
    """
    Dependency factory generating an authorization guard for specified roles.
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            logger.warning(
                f"Forbidden access: User '{current_user.username}' with role "
                f"'{current_user.role}' attempted to access restricted endpoint."
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {allowed_roles}",
            )
        return current_user

    return role_checker

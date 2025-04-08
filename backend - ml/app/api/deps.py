from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import verify_token, verify_api_key
from app.core.database import SessionLocal
from app.schemas.token import TokenPayload

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_db() -> Generator:
    """
    Database dependency
    """
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(reusable_oauth2)
) -> User:
    """
    Get current user from JWT token
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == token_data.sub).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return user

def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user

def get_api_key_user(
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[User]:
    """
    Get user from API key
    """
    if not api_key:
        return None
    
    user_id = verify_api_key(api_key)
    if not user_id:
        return None
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        return None
    
    return user

def get_current_user_from_any(
    db: Session = Depends(get_db),
    token_user: Optional[User] = Depends(get_current_user),
    api_key_user: Optional[User] = Depends(get_api_key_user)
) -> User:
    """
    Get current user from either JWT token or API key
    """
    user = token_user or api_key_user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

def check_rate_limit(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_from_any)
) -> None:
    """
    Check rate limit for current user
    """
    # TODO: Implement rate limiting using Redis
    pass

def verify_2fa(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> None:
    """
    Verify 2FA for current user
    """
    if settings.ENABLE_2FA and current_user.is_2fa_enabled and not current_user.is_2fa_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="2FA verification required"
        )

def get_current_verified_user(
    current_user: User = Depends(get_current_user),
    _: None = Depends(verify_2fa)
) -> User:
    """
    Get current user with 2FA verification
    """
    return current_user

def check_permissions(required_permissions: list[str]):
    """
    Check if current user has required permissions
    """
    def check(current_user: User = Depends(get_current_user)) -> None:
        if not any(perm in current_user.permissions for perm in required_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="The user doesn't have enough privileges"
            )
    return check

def get_current_active_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current admin user"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user

def get_current_active_analyst(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current analyst user"""
    if not current_user.is_analyst:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user

def get_current_active_viewer(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current viewer user"""
    if not current_user.is_viewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user

def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def get_current_active_user_or_none(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or None"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_anonymous(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or anonymous"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_guest(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or guest"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_public(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or public"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_private(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or private"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_restricted(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or restricted"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_limited(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or limited"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_basic(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or basic"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_premium(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or premium"""
    if not current_user or not current_user.is_active:
        return None
    return current_user

def get_current_active_user_or_enterprise(
    current_user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Get current active user or enterprise"""
    if not current_user or not current_user.is_active:
        return None
    return current_user 
from typing import Optional
from pydantic import BaseModel, EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str

class TokenPayload(BaseModel):
    sub: Optional[int] = None
    exp: Optional[int] = None
    iat: Optional[int] = None
    jti: Optional[str] = None

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []

class RefreshToken(BaseModel):
    refresh_token: str

class PasswordResetToken(BaseModel):
    token: str
    new_password: str

class EmailVerificationToken(BaseModel):
    token: str

class ApiKey(BaseModel):
    key: str
    name: str
    description: Optional[str] = None
    created_at: str
    is_active: bool
    last_used: Optional[str] = None
    expires_at: Optional[str] = None
    permissions: Optional[list[str]] = None
    rate_limit: Optional[int] = None
    ip_whitelist: Optional[list[str]] = None
    user_agent: Optional[str] = None
    metadata: Optional[dict] = None 
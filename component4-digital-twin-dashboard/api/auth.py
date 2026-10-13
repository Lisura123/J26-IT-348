import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import data_source

SECRET = os.getenv("DASHBOARD_JWT_SECRET", "dev-only-secret-change-me-before-deploying-0001")
DEMO_PASSWORD = os.getenv("DASHBOARD_DEMO_PASSWORD", "demo123")   # PP1 demo only
TOKEN_HOURS = 8

bearer = HTTPBearer()


def login(email: str, password: str) -> dict:
    u = data_source.users()
    match = u[(u.email.str.lower() == email.lower()) & (u.active.astype(str) == "True")]
    if match.empty or not hmac.compare_digest(password, DEMO_PASSWORD):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    user = match.iloc[0]
    claims = {"sub": user.user_id, "name": user.display_name, "role": user.role,
              "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)}
    return {"access_token": jwt.encode(claims, SECRET, algorithm="HS256"),
            "user": public_user(claims)}


def public_user(claims: dict) -> dict:
    perms = sorted(data_source.permissions().get(claims["role"], set()))
    return {"user_id": claims["sub"], "name": claims["name"], "role": claims["role"],
            "permissions": perms}


def current_user(cred: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        return jwt.decode(cred.credentials, SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")


def require(permission: str):
    """Dependency factory: the endpoint runs only if the user's role has `permission`."""
    def check(user: dict = Depends(current_user)) -> dict:
        if permission not in data_source.permissions().get(user["role"], set()):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Role {user['role']} lacks '{permission}'")
        return user
    return check
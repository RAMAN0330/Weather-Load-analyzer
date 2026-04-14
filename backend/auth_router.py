"""
auth_router.py — FastAPI router for authentication endpoints.

Endpoints:
  POST /auth/register  — create a new user account
  POST /auth/login     — obtain a session token
  POST /auth/logout    — invalidate the current session token
  GET  /auth/me        — return the current authenticated user
  GET  /auth/results   — list saved forecast results for current user
  GET  /auth/weights   — list saved model weights for current user
"""

import json
import os
import secrets
from datetime import timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from .auth_db import (
    SessionLocal,
    create_session,
    create_user,
    delete_session,
    get_db,
    get_session,
    get_user_by_email,
    get_user_by_id,
    get_user_by_username,
    list_forecast_results,
    list_model_weights,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer  = HTTPBearer(auto_error=False)


# ── Pydantic schemas ───────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


# ── Auth helpers ───────────────────────────────────────────────────────────
def _hash_pw(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def _verify_pw(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


def _generate_token() -> str:
    return secrets.token_urlsafe(48)


def _get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
):
    """Dependency: raises 401 if token is missing, invalid, or expired."""
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = creds.credentials
    sess = get_session(db, token)
    if not sess:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")
    from datetime import datetime
    now = datetime.now(timezone.utc)
    exp = sess.expires_at
    if exp.tzinfo is None:
        from datetime import timezone as _tz
        exp = exp.replace(tzinfo=_tz.utc)
    if now > exp:
        delete_session(db, token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    user = get_user_by_id(db, sess.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
):
    """Like _get_current_user but returns None instead of raising for unauthenticated requests."""
    if not creds:
        return None
    try:
        return _get_current_user(creds, db)
    except HTTPException:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────
@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if get_user_by_username(db, req.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    if get_user_by_email(db, req.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = create_user(db, req.username, req.email, _hash_pw(req.password))
    return user


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_username(db, req.username)
    if not user or not _verify_pw(req.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _generate_token()
    create_session(db, user.id, token)
    return {
        "token": token,
        "user": {"id": user.id, "username": user.username, "email": user.email},
    }


@router.post("/logout")
def logout(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
):
    if creds:
        delete_session(db, creds.credentials)
    return {"detail": "Logged out"}


@router.get("/me", response_model=UserOut)
def me(user=Depends(_get_current_user)):
    return user


@router.get("/results")
def saved_results(
    limit: int = 50,
    user=Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    rows = list_forecast_results(db, user_id=user.id, limit=limit)
    return [
        {
            "id": r.id,
            "job_id": r.job_id,
            "date": r.date,
            "region": r.region,
            "baseline_days": r.baseline_days,
            "created_at": r.created_at.isoformat(),
            "result": json.loads(r.result_json),
        }
        for r in rows
    ]


@router.get("/weights")
def saved_weights(
    region: Optional[str] = None,
    limit: int = 50,
    user=Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    rows = list_model_weights(db, user_id=user.id, region=region, limit=limit)
    return [
        {
            "id": r.id,
            "region": r.region,
            "weights_type": r.weights_type,
            "created_at": r.created_at.isoformat(),
            "weights": json.loads(r.weights_json),
        }
        for r in rows
    ]

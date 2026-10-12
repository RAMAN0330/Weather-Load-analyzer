"""Bearer-token authentication for the FastAPI service.

Users log in at ``/auth/login`` (``auth_router``), which stores a session token
in PostgreSQL. This middleware accepts a request when:

* the path is public (``/auth/login``, ``/auth/register``), or
* the token is the shared service key ``API_SECRET_KEY`` (internal callers), or
* the token is a live, unexpired session in the ``user_sessions`` table.

Successful lookups are cached briefly so each request does not cost a query.
Browsers cannot set headers on a WebSocket, so WebSocket clients pass the token
as a ``?token=`` query parameter instead.

Set ``FASTAPI_AUTH_DISABLED=true`` only for local development without a database.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from typing import Dict, Optional
from urllib.parse import parse_qs

from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_CACHE_TTL_S = float(os.getenv("FASTAPI_AUTH_CACHE_TTL", "60"))

# token -> monotonic expiry time
_valid_tokens: Dict[str, float] = {}


def _auth_disabled() -> bool:
    return os.getenv("FASTAPI_AUTH_DISABLED", "").strip().lower() in ("1", "true", "yes")


PUBLIC_PATHS = frozenset({"/auth/login", "/auth/register"})


def _is_service_token(token: str) -> bool:
    secret = os.getenv("API_SECRET_KEY", "")
    return bool(secret) and hmac.compare_digest(token, secret)


def _check_session(token: str) -> bool:
    """True if ``token`` is an unexpired session in the database."""
    try:
        try:
            from . import auth_db
        except ImportError:
            import auth_db  # type: ignore
        with auth_db.SessionLocal() as db:
            sess = auth_db.get_session(db, token)
            if sess is None:
                return False
            expires = sess.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            return expires > datetime.now(timezone.utc)
    except Exception as exc:  # database unreachable
        logger.warning("Session lookup failed: %s", exc)
        return False


async def is_token_valid(token: Optional[str]) -> bool:
    if not token:
        return False
    if _is_service_token(token):
        return True
    now = time.monotonic()
    expiry = _valid_tokens.get(token)
    if expiry is not None and expiry > now:
        return True
    ok = await asyncio.to_thread(_check_session, token)
    if ok:
        if len(_valid_tokens) > 10_000:
            for t, exp in list(_valid_tokens.items()):
                if exp <= now:
                    del _valid_tokens[t]
        _valid_tokens[token] = now + _CACHE_TTL_S
    else:
        _valid_tokens.pop(token, None)
    return ok


def _bearer_from_headers(scope) -> Optional[str]:
    for name, value in scope.get("headers") or []:
        if name == b"authorization":
            text = value.decode("latin-1")
            if text.startswith("Bearer "):
                return text[7:].strip() or None
    return None


def _token_from_query(scope) -> Optional[str]:
    qs = parse_qs((scope.get("query_string") or b"").decode("latin-1"))
    values = qs.get("token") or []
    return values[0] if values else None


class BearerAuthMiddleware:
    """Pure ASGI middleware; register it *before* CORSMiddleware so CORS stays
    outermost and 401 responses still carry CORS headers."""

    def __init__(self, app):
        self.app = app
        if _auth_disabled():
            logger.warning("FASTAPI_AUTH_DISABLED is set: all API routes are unauthenticated.")

    async def __call__(self, scope, receive, send):
        scope_type = scope["type"]
        if scope_type not in ("http", "websocket") or _auth_disabled():
            await self.app(scope, receive, send)
            return

        if scope_type == "http":
            if scope.get("method") == "OPTIONS" or scope.get("path") in PUBLIC_PATHS:
                await self.app(scope, receive, send)
                return
            if await is_token_valid(_bearer_from_headers(scope)):
                await self.app(scope, receive, send)
                return
            body = json.dumps({"detail": "Not authenticated"}).encode()
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        token = _bearer_from_headers(scope) or _token_from_query(scope)
        if await is_token_valid(token):
            await self.app(scope, receive, send)
            return
        # Reject the handshake; 1008 = policy violation.
        await receive()  # websocket.connect
        await send({"type": "websocket.close", "code": 1008})

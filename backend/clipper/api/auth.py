"""Owner sign-in, server-side sessions, CSRF tokens and login throttling.

* Password: argon2 hash from ``CLIPPER_OWNER_PASSWORD_HASH`` (argon2-cffi).
* Session cookie ``clipper_session``: random token; only HMAC-SHA256(session_secret, token)
  is stored. HttpOnly, SameSite=Lax, Path=/, host-only (no Domain), Secure when configured.
* Expiry = min(created + TTL, last_seen + idle), enforced server-side; last_seen slides
  (written at most once a minute).
* CSRF: every session has its own token; with auth disabled (loopback only) a random
  local token stored in ``meta.local_csrf`` is used. Mutations must send ``X-CSRF-Token``.
* Throttling: failures per IP within the window and a global cap → 429 + Retry-After.
  Each attempt is checked and recorded atomically before the password is verified, so
  concurrent requests cannot exceed either limit.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import sqlite3
import time
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import APIRouter, Depends, Request, Response

from .. import db
from ..config import Settings
from .deps import SAFE_METHODS, ApiError, client_ip, get_conn, get_settings
from .schemas import LoginIn, SessionOut

SESSION_COOKIE = "clipper_session"
CSRF_HEADER = "X-CSRF-Token"
LAST_SEEN_WRITE_INTERVAL = 60.0       # seconds between sliding last_seen updates
GLOBAL_FAILURE_CAP = 50               # failed logins (all IPs) per window
PURGE_INTERVAL = 600.0                # seconds between expired-row purges per database

_hasher = PasswordHasher()
_last_purge: dict[str, float] = {}


def now() -> float:
    """Wall clock (seconds). Tests monkeypatch this to simulate expiry."""
    return time.time()


# ----------------------------------------------------------------------------- tokens
def _secret(settings: Settings) -> bytes:
    if settings.session_secret is None:
        raise RuntimeError("session secret not configured")
    return settings.session_secret.get_secret_value().encode("utf-8")


def hash_token(settings: Settings, token: str) -> str:
    return hmac.new(_secret(settings), token.encode("utf-8"), hashlib.sha256).hexdigest()


def local_csrf_token(conn: sqlite3.Connection) -> str:
    """The CSRF token used while auth is disabled (created once, kept in ``meta``)."""
    row = conn.execute("SELECT value FROM meta WHERE key='local_csrf'").fetchone()
    if row is not None and row["value"]:
        return row["value"]
    token = secrets.token_urlsafe(32)
    conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('local_csrf', ?)", (token,))
    return conn.execute("SELECT value FROM meta WHERE key='local_csrf'").fetchone()["value"]


# ----------------------------------------------------------------------------- credentials
def verify_credentials(settings: Settings, username: str, password: str) -> bool:
    """Constant-time username compare; the password hash is always checked."""
    expected = settings.owner_username.encode("utf-8")
    user_ok = hmac.compare_digest(username.encode("utf-8"), expected)
    stored = settings.owner_password_hash.get_secret_value() if settings.owner_password_hash else ""
    try:
        pw_ok = _hasher.verify(stored, password)
    except (VerificationError, InvalidHashError):
        pw_ok = False
    return bool(user_ok and pw_ok)


# ----------------------------------------------------------------------------- sessions
@dataclass(frozen=True)
class SessionInfo:
    token_hash: str
    username: str
    csrf_token: str
    expires_at: float          # effective expiry (absolute TTL or idle timeout, whichever is first)


def _ttl(settings: Settings) -> float:
    return settings.session_ttl_hours * 3600.0


def _idle(settings: Settings) -> float:
    return settings.session_idle_minutes * 60.0


def _effective_expiry(row: sqlite3.Row, settings: Settings) -> float:
    return min(float(row["expires_at"]), float(row["created_at"]) + _ttl(settings),
               float(row["last_seen_at"]) + _idle(settings))


def create_session(conn: sqlite3.Connection, settings: Settings, username: str) -> tuple[str, SessionInfo]:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)
    t = now()
    th = hash_token(settings, token)
    conn.execute(
        """INSERT INTO sessions(token_hash, csrf_token, username, created_at, last_seen_at, expires_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (th, csrf, username, t, t, t + _ttl(settings)),
    )
    return token, SessionInfo(th, username, csrf, min(t + _ttl(settings), t + _idle(settings)))


def lookup_session(conn: sqlite3.Connection, settings: Settings, token: str | None) -> SessionInfo | None:
    """Valid session for ``token`` (sliding last_seen), else None. Expired rows are deleted."""
    if not token or len(token) > 256:
        return None
    th = hash_token(settings, token)
    row = conn.execute("SELECT * FROM sessions WHERE token_hash=?", (th,)).fetchone()
    t = now()
    _maybe_purge(conn, settings, t)
    if row is None:
        return None
    if t >= _effective_expiry(row, settings):
        conn.execute("DELETE FROM sessions WHERE token_hash=?", (th,))
        return None
    last_seen = float(row["last_seen_at"])
    if t - last_seen >= LAST_SEEN_WRITE_INTERVAL:
        conn.execute("UPDATE sessions SET last_seen_at=? WHERE token_hash=?", (t, th))
        last_seen = t
    expiry = min(float(row["expires_at"]), float(row["created_at"]) + _ttl(settings), last_seen + _idle(settings))
    return SessionInfo(th, row["username"], row["csrf_token"], expiry)


def delete_session(conn: sqlite3.Connection, token_hash: str) -> None:
    conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))


def purge_expired(conn: sqlite3.Connection, settings: Settings, at: float | None = None) -> None:
    t = now() if at is None else at
    conn.execute("DELETE FROM sessions WHERE expires_at <= ? OR last_seen_at <= ?", (t, t - _idle(settings)))
    conn.execute("DELETE FROM login_attempts WHERE at < ?", (t - settings.login_window_seconds,))


def _maybe_purge(conn: sqlite3.Connection, settings: Settings, t: float) -> None:
    key = str(settings.db_path)
    if t - _last_purge.get(key, 0.0) >= PURGE_INTERVAL:
        _last_purge[key] = t
        purge_expired(conn, settings, t)


# ----------------------------------------------------------------------------- throttling
def _retry_after(times: list[float], limit: int, window: float, t: float) -> int | None:
    """Seconds until fewer than ``limit`` failures remain in the window, or None."""
    if len(times) < limit:
        return None
    release_at = times[len(times) - limit] + window
    return max(1, math.ceil(release_at - t))


def login_retry_after(conn: sqlite3.Connection, settings: Settings, ip: str) -> int | None:
    t = now()
    since = t - settings.login_window_seconds
    per_ip = [r[0] for r in conn.execute(
        "SELECT at FROM login_attempts WHERE ip=? AND success=0 AND at>? ORDER BY at", (ip, since))]
    everyone = [r[0] for r in conn.execute(
        "SELECT at FROM login_attempts WHERE success=0 AND at>? ORDER BY at", (since,))]
    waits = [w for w in (
        _retry_after(per_ip, max(1, settings.login_max_failures), settings.login_window_seconds, t),
        _retry_after(everyone, GLOBAL_FAILURE_CAP, settings.login_window_seconds, t),
    ) if w is not None]
    return max(waits) if waits else None


def record_failure(conn: sqlite3.Connection, ip: str) -> None:
    conn.execute("INSERT INTO login_attempts(ip, at, success) VALUES (?, ?, 0)", (ip, now()))


def clear_failures(conn: sqlite3.Connection, ip: str) -> None:
    conn.execute("DELETE FROM login_attempts WHERE ip=?", (ip,))


# ----------------------------------------------------------------------------- request state
@dataclass(frozen=True)
class AuthState:
    authenticated: bool
    auth_required: bool
    session: SessionInfo | None = None
    csrf_token: str | None = None


def get_auth_state(request: Request, conn: sqlite3.Connection = Depends(get_conn),
                   settings: Settings = Depends(get_settings)) -> AuthState:
    if not settings.auth_enabled:
        return AuthState(True, False, None, request.app.state.local_csrf)
    info = lookup_session(conn, settings, request.cookies.get(SESSION_COOKIE))
    if info is None:
        return AuthState(False, True)
    return AuthState(True, True, info, info.csrf_token)


def require_auth(state: AuthState = Depends(get_auth_state)) -> AuthState:
    if not state.authenticated:
        raise ApiError(401, "Please sign in to continue.", "auth_required")
    return state


def check_csrf(request: Request, state: AuthState) -> None:
    if request.method.upper() in SAFE_METHODS:
        return
    expected = state.csrf_token
    provided = request.headers.get(CSRF_HEADER, "")
    if not expected or not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise ApiError(403, "The security token is missing or out of date. Reload the page and try again.",
                       "csrf_failed")


def require_csrf(request: Request, state: AuthState = Depends(get_auth_state)) -> None:
    check_csrf(request, state)


# ----------------------------------------------------------------------------- responses
def session_out(settings: Settings, state: AuthState) -> SessionOut:
    if not settings.auth_enabled:
        return SessionOut(authenticated=True, auth_required=False, mode=settings.mode,
                          username=None, csrf_token=state.csrf_token, expires_at=None)
    if state.session is None:
        return SessionOut(authenticated=False, auth_required=True, mode=settings.mode)
    return SessionOut(authenticated=True, auth_required=True, mode=settings.mode,
                      username=state.session.username, csrf_token=state.session.csrf_token,
                      expires_at=state.session.expires_at)


def _set_cookie(response: Response, settings: Settings, token: str) -> None:
    response.set_cookie(SESSION_COOKIE, token, max_age=int(_ttl(settings)), path="/", domain=None,
                        secure=settings.secure_cookies, httponly=True, samesite="Lax")


def _clear_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", domain=None, secure=settings.secure_cookies,
                           httponly=True, samesite="Lax")


# ----------------------------------------------------------------------------- routes
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/session", response_model=SessionOut)
def get_session(settings: Settings = Depends(get_settings),
                state: AuthState = Depends(get_auth_state)) -> SessionOut:
    return session_out(settings, state)


@router.post("/login", response_model=SessionOut)
def login(body: LoginIn, request: Request, response: Response,
          conn: sqlite3.Connection = Depends(get_conn),
          settings: Settings = Depends(get_settings),
          state: AuthState = Depends(get_auth_state)) -> SessionOut:
    if not settings.auth_enabled:
        return session_out(settings, state)
    ip = client_ip(request, settings)
    # Check and record in one write transaction: the attempt is counted as a failure *before*
    # the slow password check, so concurrent requests see each other and cannot all slip
    # under the per-IP limit or the global cap. A successful sign-in clears it again below.
    with db.tx(conn):
        wait = login_retry_after(conn, settings, ip)
        if wait is None:
            record_failure(conn, ip)
    if wait is not None:
        minutes = max(1, math.ceil(wait / 60))
        raise ApiError(429, f"Too many failed sign-in attempts. Try again in about {minutes} minute"
                            f"{'s' if minutes != 1 else ''}.", "rate_limited", headers={"Retry-After": str(wait)})
    if not verify_credentials(settings, body.username, body.password):
        raise ApiError(401, "Wrong username or password.", "invalid_credentials")   # failure already recorded
    with db.tx(conn):
        clear_failures(conn, ip)
        if state.session is not None:            # no session fixation: always issue a fresh token
            delete_session(conn, state.session.token_hash)
        token, info = create_session(conn, settings, settings.owner_username)
        purge_expired(conn, settings)
    _set_cookie(response, settings, token)
    return session_out(settings, AuthState(True, True, info, info.csrf_token))


@router.post("/logout", response_model=SessionOut)
def logout(request: Request, response: Response,
           conn: sqlite3.Connection = Depends(get_conn),
           settings: Settings = Depends(get_settings),
           state: AuthState = Depends(get_auth_state)) -> SessionOut:
    if not settings.auth_enabled:
        check_csrf(request, state)
        return session_out(settings, state)
    if state.session is not None:
        check_csrf(request, state)
        delete_session(conn, state.session.token_hash)
    _clear_cookie(response, settings)
    return SessionOut(authenticated=False, auth_required=True, mode=settings.mode)

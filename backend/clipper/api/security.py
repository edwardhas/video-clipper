"""Transport-level protections applied to every HTTP request.

* Host header allow-list (DNS-rebinding defence) → 400.
* ``Sec-Fetch-Site: cross-site`` requests to /api and /media → 403.
* Non-safe methods need an allowed ``Origin`` (or, if absent, a ``Referer`` from an
  allowed origin) → 403. The session-bound ``X-CSRF-Token`` check lives in
  :mod:`clipper.api.auth` because it needs the session.
* JSON request bodies are size-limited (uploads have their own streaming limit).
* Security headers on every response; ``Cache-Control: no-store`` for /api.
* Unhandled exceptions become a generic JSON 500 (no paths, no tracebacks).

No CORS headers are ever emitted.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..config import Settings
from .deps import SAFE_METHODS

log = logging.getLogger("clipper.api")

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
    "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
JSON_BODY_LIMIT = 1024 * 1024          # bytes, for every non-upload request body
UPLOAD_PATH = "/api/uploads"


def host_from_header(value: str) -> str:
    """Hostname part of a Host header (port removed, IPv6 brackets kept), lower-case."""
    value = (value or "").strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[: end + 1] if end != -1 else value
    host, sep, port = value.rpartition(":")
    if sep and port.isdigit():
        return host
    return value


def host_allowed(host_header: str | None, allowed: set[str]) -> bool:
    if not host_header:
        return False
    return host_from_header(host_header) in allowed


def _origin_of(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}".lower()


def origin_allowed(origin: str | None, referer: str | None, allowed: set[str]) -> bool:
    """Origin must match exactly; without Origin, the Referer's origin must match."""
    if origin is not None:
        return origin.strip().rstrip("/").lower() in allowed
    if referer:
        ref_origin = _origin_of(referer.strip())
        return ref_origin is not None and ref_origin in allowed
    return False


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _json_body(detail: str, code: str) -> bytes:
    return json.dumps({"detail": detail, "code": code}).encode("utf-8")


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self.app = app
        self.allowed_hosts = {h.lower() for h in settings.allowed_host_list}
        self.allowed_origins = {o.lower() for o in settings.allowed_origin_list}

    # -- checks -----------------------------------------------------------------
    def _reject_reason(self, scope: Scope, headers: Headers) -> tuple[int, str, str] | None:
        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET").upper()
        if not host_allowed(headers.get("host"), self.allowed_hosts):
            return 400, "This address is not allowed for this app.", "bad_host"
        api_or_media = _is_under(path, "/api") or _is_under(path, "/media")
        if api_or_media and headers.get("sec-fetch-site", "").lower() == "cross-site":
            return 403, "Cross-site requests are not allowed.", "cross_site"
        if method not in SAFE_METHODS:
            if not origin_allowed(headers.get("origin"), headers.get("referer"), self.allowed_origins):
                return 403, "This request did not come from the app's own page.", "bad_origin"
            if path != UPLOAD_PATH:
                length = headers.get("content-length")
                if length is not None and (not length.isdigit() or int(length) > JSON_BODY_LIMIT):
                    return 413, "The request is too large.", "too_large"
        return None

    # -- ASGI -------------------------------------------------------------------
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        is_api = _is_under(path, "/api")
        is_media = _is_under(path, "/media")
        started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                hdrs = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    hdrs[name] = value
                if is_api:
                    hdrs["Cache-Control"] = "no-store"
                elif is_media and "cache-control" not in hdrs:
                    hdrs["Cache-Control"] = "private, no-cache"
            await send(message)

        headers = Headers(scope=scope)
        reason = self._reject_reason(scope, headers)
        if reason is not None:
            await self._send_json(send_wrapper, *reason)
            return

        if path != UPLOAD_PATH and scope.get("method", "GET").upper() not in SAFE_METHODS:
            receive = _limited_receive(receive, JSON_BODY_LIMIT)
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            if started:
                raise
            log.exception("Unhandled error while serving %s %s", scope.get("method"), path)
            await self._send_json(send_wrapper, 500,
                                  "Something went wrong on the server. Please try again.", "internal_error")

    @staticmethod
    async def _send_json(send: Send, status: int, detail: str, code: str) -> None:
        body = _json_body(detail, code)
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})


def _limited_receive(receive: Receive, limit: int) -> Receive:
    total = 0

    async def wrapped() -> Message:
        nonlocal total
        message = await receive()
        if message["type"] == "http.request":
            total += len(message.get("body", b""))
            if total > limit:
                raise HTTPException(status_code=413, detail="The request is too large.")
        return message

    return wrapped

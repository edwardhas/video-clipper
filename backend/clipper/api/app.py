"""FastAPI application factory: ``uvicorn --factory clipper.api.app:create_app``.

Serves the JSON API under ``/api``, media files under ``/media`` and the built
frontend (``frontend/dist``) for every other GET path (SPA fallback). Errors are
always JSON ``{"detail", "code"}`` with plain-language messages; paths, secrets
and tracebacks never reach the client.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__, db
from ..config import Settings, get_settings
from ..errors import (
    CancelledError,
    ClipperError,
    DiskSpaceError,
    InputError,
    MediaError,
    MissingDependencyError,
    NoSpeechError,
    ProviderNotConfiguredError,
    SourceUnavailableError,
)
from ..storage import Storage
from . import auth, routes_clips, routes_media, routes_projects, routes_system, routes_uploads
from .deps import ApiError
from .security import SecurityMiddleware

log = logging.getLogger("clipper.api")

# Native OpenTelemetry in FastAPI would see request bodies (incl. passwords): keep it off.
TELEMETRY_OFF: dict[str, Any] = {
    "tracing": False, "metrics": False, "logs": False, "operation_spans": False, "auto_configure": False,
}
GENERIC_500 = "Something went wrong on the server. Please try again."
_STATUS_CODES = {
    400: "bad_request", 401: "auth_required", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
    409: "conflict", 411: "length_required", 413: "too_large", 415: "unsupported_media_type",
    422: "invalid_input", 429: "rate_limited",
}
_STATUS_MESSAGES = {
    404: "This page or item was not found.",
    405: "This action is not supported here.",
}

NOT_BUILT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local Clipper</title>
<style>body{font-family:system-ui,sans-serif;background:#F7F1F4;color:#2b1026;margin:0;padding:3rem 1.5rem}
main{max-width:36rem;margin:0 auto}h1{color:#591E4A}code{background:#fff;padding:.1rem .3rem;border-radius:4px}</style>
</head><body><main><h1>Local Clipper</h1>
<p>The server is running, but the web interface has not been built yet.</p>
<p>Run <code>scripts/setup.sh</code> (it builds the frontend), then reload this page.</p>
</main></body></html>
"""


# ----------------------------------------------------------------------------- error mapping
def _json_error(status: int, detail: str, code: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail, "code": code}, status_code=status, headers=headers)


def clipper_error_status(exc: ClipperError) -> int:
    if isinstance(exc, DiskSpaceError):
        return 507
    if isinstance(exc, (MissingDependencyError, ProviderNotConfiguredError)):
        return 503
    if isinstance(exc, (InputError, NoSpeechError, MediaError, SourceUnavailableError)):
        return 422
    if isinstance(exc, CancelledError):
        return 409
    return 400


def validation_message(errors: list[dict[str, Any]]) -> str:
    """Readable summary of request validation errors (never echoes submitted values)."""
    parts: list[str] = []
    for err in errors[:5]:
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path", "header")]
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{'.'.join(loc)}: {msg}" if loc else msg)
    return "Some values were not accepted: " + "; ".join(parts) if parts else "The request was not valid."


async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return _json_error(exc.status_code, exc.detail, exc.code, exc.headers)


async def _clipper_error_handler(_: Request, exc: ClipperError) -> JSONResponse:
    return _json_error(clipper_error_status(exc), exc.user_message, exc.code)


async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _json_error(422, validation_message(list(exc.errors())), "invalid_input")


async def _http_error_handler(_: Request, exc: StarletteHTTPException) -> Response:
    detail = exc.detail if isinstance(exc.detail, str) else ""
    if exc.status_code in _STATUS_MESSAGES or not detail or detail in ("Not Found", "Method Not Allowed"):
        detail = _STATUS_MESSAGES.get(exc.status_code, detail or "The request could not be completed.")
    code = _STATUS_CODES.get(exc.status_code, "error")
    return _json_error(exc.status_code, detail, code, getattr(exc, "headers", None))


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled error while serving %s %s", request.method, request.url.path, exc_info=exc)
    return _json_error(500, GENERIC_500, "internal_error")


# ----------------------------------------------------------------------------- frontend
def _is_backend_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in ("api", "media"))


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    dist = dist.resolve()
    index = dist / "index.html"
    if not index.is_file():
        @app.get("/{full_path:path}", include_in_schema=False, response_class=HTMLResponse)
        def frontend_not_built(full_path: str) -> Response:
            if _is_backend_path(full_path):
                raise StarletteHTTPException(404)
            return HTMLResponse(NOT_BUILT_HTML, headers={"Cache-Control": "no-cache"})
        return

    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> Response:
        if _is_backend_path(full_path):
            raise StarletteHTTPException(404)
        hidden = any(part.startswith(".") for part in full_path.split("/"))
        if full_path and not hidden and "\x00" not in full_path:
            candidate = (dist / full_path).resolve()
            if candidate.is_relative_to(dist) and candidate.is_file() and candidate != index:
                return FileResponse(candidate)
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})


# ----------------------------------------------------------------------------- factory
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_for_startup()
    storage = Storage(settings.data_dir)
    storage.ensure()
    db.init_db(settings.db_path)
    with db.session(settings.db_path) as conn:
        local_csrf = auth.local_csrf_token(conn)

    app = FastAPI(title="Local Clipper", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url=None, telemetry=TELEMETRY_OFF)  # type: ignore[arg-type]
    app.state.settings = settings
    app.state.storage = storage
    app.state.local_csrf = local_csrf

    app.add_exception_handler(ApiError, _api_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ClipperError, _clipper_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_handler)

    app.include_router(routes_system.public_router)
    app.include_router(auth.router)
    app.include_router(routes_system.router)
    app.include_router(routes_uploads.router)
    app.include_router(routes_projects.router)
    app.include_router(routes_clips.router)
    app.include_router(routes_media.router)

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                   include_in_schema=False)
    def api_not_found(rest: str) -> Response:
        raise StarletteHTTPException(404)

    _mount_frontend(app, settings.frontend_dist)
    app.add_middleware(SecurityMiddleware, settings=settings)
    return app

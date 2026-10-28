"""Shared FastAPI dependencies and small HTTP helpers for the API routers."""

from __future__ import annotations

import ipaddress
import sqlite3
from typing import Iterator

from fastapi import Request

from .. import db
from ..config import Settings
from ..models import ProjectOptions
from ..storage import Storage, is_valid_id
from .schemas import DefaultsModel

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


class ApiError(Exception):
    """An HTTP error with a plain-language message and a stable code."""

    def __init__(self, status_code: int, detail: str, code: str,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.headers = headers


def not_found(what: str = "That item") -> ApiError:
    return ApiError(404, f"{what} was not found.", "not_found")


def conflict(detail: str) -> ApiError:
    return ApiError(409, detail, "conflict")


def invalid(detail: str) -> ApiError:
    return ApiError(422, detail, "invalid_input")


# ----------------------------------------------------------------------------- app state
def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    """One short-lived SQLite connection per request, always closed afterwards."""
    conn = db.connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def require_id(value: str, what: str) -> str:
    """Reject anything that is not an app-generated id (blocks traversal attempts)."""
    if not is_valid_id(value):
        raise not_found(what)
    return value


# ----------------------------------------------------------------------------- defaults
def defaults_from_settings(settings: Settings) -> DefaultsModel:
    try:
        return DefaultsModel(
            clip_count=min(settings.default_clip_count, settings.max_clips),
            min_seconds=settings.default_min_seconds,
            max_seconds=settings.default_max_seconds,
            captions_enabled=True,
        )
    except ValueError:
        return DefaultsModel()


def load_defaults(conn: sqlite3.Connection, settings: Settings) -> DefaultsModel:
    """Saved defaults (settings table key 'defaults'), else values from configuration."""
    row = conn.execute("SELECT value FROM settings WHERE key='defaults'").fetchone()
    if row is not None:
        try:
            return DefaultsModel.model_validate_json(row["value"])
        except ValueError:
            pass
    return defaults_from_settings(settings)


def options_from_defaults(defaults: DefaultsModel) -> ProjectOptions:
    return ProjectOptions(
        clip_count=defaults.clip_count,
        min_seconds=defaults.min_seconds,
        max_seconds=defaults.max_seconds,
        captions_enabled=defaults.captions_enabled,
    )


# ----------------------------------------------------------------------------- client address
def _parse_networks(spec: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    nets = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            continue
    return nets


def _in_networks(addr: str, nets: list) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return any(ip in n for n in nets)


def client_ip(request: Request, settings: Settings) -> str:
    """Peer address; X-Forwarded-For is honoured only from configured trusted proxies.

    Applies in both modes: a local-mode app behind a same-host proxy or tunnel sees every client
    as 127.0.0.1, so without this the login throttle would be one bucket anyone can fill. Local
    mode runs uvicorn with --no-proxy-headers, so the peer here is the raw TCP peer and the
    rightmost untrusted hop (the one the proxy appended or wrote) is used.
    """
    peer = request.client.host if request.client else "unknown"
    nets = _parse_networks(settings.trusted_proxies)
    if not nets or not _in_networks(peer, nets):
        return peer
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    for hop in reversed(hops):
        if not _in_networks(hop, nets):
            return hop[:64]
    return peer

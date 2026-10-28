"""Health, system information and saved defaults.

* ``GET /api/health`` – liveness only (no auth, no details).
* ``GET /api/system`` – storage, OpenAI disclosure, limits, tool versions, worker status.
* ``GET/PUT /api/settings`` – owner defaults (``DefaultsModel``) in the ``settings`` table.

Tool versions are probed lazily (other modules may be missing while the app is
being built) and cached briefly so the settings page stays fast.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends

from .. import __version__, db
from ..config import Settings
from ..models import ProjectOptions
from ..storage import Storage
from .auth import require_auth, require_csrf
from .deps import get_conn, get_settings, get_storage, invalid, load_defaults
from .schemas import (
    DefaultsModel,
    LimitsInfo,
    OpenAIInfo,
    StorageInfo,
    SystemOut,
    ToolsInfo,
    WorkerInfo,
)

log = logging.getLogger("clipper.api")

MB = 1024 * 1024
WORKER_ONLINE_SECONDS = 30.0
TOOLS_CACHE_SECONDS = 300.0
USAGE_CACHE_SECONDS = 30.0
MIN_SECONDS_FLOOR = 5.0          # ProjectOptions.min_seconds lower bound
MAX_SECONDS_CAP = 180.0          # ProjectOptions.max_seconds upper bound
WHISPER_REMOVAL_DATE = "2027-02-26"

DATA_SENT: list[str] = [
    "Extracted audio (a compressed speech track, sent in chunks) is sent to OpenAI for transcription.",
    "The transcript text is sent to OpenAI to find and rank the best moments.",
    "A few still frames per candidate clip are sent to OpenAI to review the shortlisted clips.",
    "Your source videos and rendered clips are not uploaded to OpenAI; they stay in this app's storage.",
]

public_router = APIRouter(tags=["system"])
router = APIRouter(prefix="/api", tags=["system"], dependencies=[Depends(require_auth), Depends(require_csrf)])


class _TimedCache:
    """Tiny thread-safe value cache with a time-to-live (per key)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, ttl: float, compute: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            hit = self._values.get(key)
            if hit is not None and now - hit[0] < ttl:
                return hit[1]
        value = compute()
        with self._lock:
            self._values[key] = (time.monotonic(), value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


_cache = _TimedCache()


# ----------------------------------------------------------------------------- tools
def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # a broken/missing optional module must not break the page
        log.warning("system info probe failed", exc_info=True)
        return None


def _ffmpeg_version(settings: Settings) -> str | None:
    from ..media.probe import tool_version
    line = tool_version(settings.ffmpeg)
    if not line:
        return None
    match = re.search(r"version\s+(\S+)", line)
    return match.group(1)[:60] if match else line[:60]


def _ytdlp_version() -> str | None:
    from ..media.youtube import ytdlp_version
    return ytdlp_version()


def _js_runtime(settings: Settings) -> str | None:
    from ..media.youtube import js_runtime_info
    return js_runtime_info(settings)


def _face_detector() -> str | None:
    from ..framing.faces import detector_available
    return "YuNet (OpenCV)" if detector_available() else None


def _static_tools(settings: Settings) -> dict[str, str | None]:
    return {
        "ffmpeg_version": _safe(lambda: _ffmpeg_version(settings)),
        "ytdlp_version": _safe(_ytdlp_version),
        "js_runtime": _safe(lambda: _js_runtime(settings)),
        "face_detector": _safe(_face_detector),
    }


# ----------------------------------------------------------------------------- worker
def worker_status(conn: sqlite3.Connection, now: float | None = None) -> tuple[WorkerInfo, str | None]:
    """Worker liveness from the freshest heartbeat, and the encoder of the newest worker."""
    t = db.now() if now is None else now
    latest = conn.execute("SELECT MAX(heartbeat_at) FROM workers").fetchone()[0]
    age = None if not latest else max(0.0, t - float(latest))
    info = WorkerInfo(online=age is not None and age < WORKER_ONLINE_SECONDS,
                      last_heartbeat_age_s=None if age is None else round(age, 1))
    encoder: str | None = None
    for row in conn.execute("SELECT info FROM workers ORDER BY started_at DESC LIMIT 5"):
        data = db.loads(row["info"], {}) or {}
        if isinstance(data, dict) and isinstance(data.get("encoder"), str):
            encoder = data["encoder"][:60]
            break
    return info, encoder


# ----------------------------------------------------------------------------- storage/openai
def data_dir_display(settings: Settings, storage: Storage) -> str:
    """Absolute path on the owner's computer; never the host path in hosted mode."""
    if settings.mode == "local":
        return str(storage.root)
    return "/data (server volume)" if storage.root == Path("/data") else "App data volume (server)"


def storage_info(settings: Settings, storage: Storage) -> StorageInfo:
    used = _cache.get(f"usage:{storage.root}", USAGE_CACHE_SECONDS, storage.usage_bytes)
    return StorageInfo(
        location_label="This computer" if settings.mode == "local" else "Server data volume",
        data_dir_display=data_dir_display(settings, storage),
        free_bytes=int(storage.free_bytes()),
        used_bytes=int(used),
        reserve_bytes=int(settings.min_free_disk_mb) * MB,
    )


def transcription_note(settings: Settings) -> str:
    if settings.transcribe_model == "whisper-1":
        return ("Transcription uses whisper-1, the only OpenAI transcription model that returns word "
                f"timestamps. OpenAI has announced whisper-1 will be removed on {WHISPER_REMOVAL_DATE}; "
                "its replacement does not return word timestamps, so an update of this app will be "
                "needed before then.")
    return (f"Transcription uses {settings.transcribe_model}. Word-level timestamps are required for "
            "captions; check that this model provides them.")


def openai_info(settings: Settings) -> OpenAIInfo:
    return OpenAIInfo(
        configured=settings.openai_configured,
        transcribe_model=settings.transcribe_model,
        select_model=settings.select_model,
        review_model=settings.review_model,
        data_sent=list(DATA_SENT),
        transcription_note=transcription_note(settings),
    )


def limits_info(settings: Settings) -> LimitsInfo:
    return LimitsInfo(max_source_minutes=settings.max_source_minutes, max_upload_mb=settings.max_upload_mb,
                      max_clips=settings.max_clips, min_seconds_floor=MIN_SECONDS_FLOOR,
                      max_seconds_cap=MAX_SECONDS_CAP)


# ----------------------------------------------------------------------------- routes
@public_router.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/system", response_model=SystemOut)
def system(conn: sqlite3.Connection = Depends(get_conn),
           settings: Settings = Depends(get_settings),
           storage: Storage = Depends(get_storage)) -> SystemOut:
    tools = _cache.get(f"tools:{settings.ffmpeg}:{settings.ytdlp_js_runtime}", TOOLS_CACHE_SECONDS,
                       lambda: _static_tools(settings))
    worker, encoder = worker_status(conn)
    return SystemOut(
        mode=settings.mode,
        version=__version__,
        public_origin=settings.effective_public_origin,
        storage=storage_info(settings, storage),
        openai=openai_info(settings),
        limits=limits_info(settings),
        tools=ToolsInfo(encoder=encoder, **tools),
        worker=worker,
    )


@router.get("/settings", response_model=DefaultsModel)
def get_defaults(conn: sqlite3.Connection = Depends(get_conn),
                 settings: Settings = Depends(get_settings)) -> DefaultsModel:
    return load_defaults(conn, settings)


@router.put("/settings", response_model=DefaultsModel)
def put_defaults(body: DefaultsModel,
                 conn: sqlite3.Connection = Depends(get_conn),
                 settings: Settings = Depends(get_settings)) -> DefaultsModel:
    if body.clip_count > settings.max_clips:
        raise invalid(f"At most {settings.max_clips} clips per video are allowed.")
    try:
        ProjectOptions(clip_count=body.clip_count, min_seconds=body.min_seconds, max_seconds=body.max_seconds,
                       captions_enabled=body.captions_enabled)
    except ValueError:
        raise invalid("The maximum clip length must be at least 5 seconds above the minimum.") from None
    conn.execute("INSERT INTO settings(key, value) VALUES ('defaults', ?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (body.model_dump_json(),))
    return body

"""POST /api/uploads – stream a raw request body into ``uploads/<id>/`` with hard limits."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import unquote

import anyio
from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from .. import db
from ..config import Settings
from ..storage import Storage, new_id, safe_filename
from .auth import require_auth, require_csrf
from .deps import ApiError, get_conn, get_settings, get_storage, invalid
from .schemas import UploadOut

ALLOWED_EXTENSIONS = frozenset({
    "mp4", "mov", "m4v", "mkv", "webm", "avi", "mpg", "mpeg", "ts", "mts", "m2ts", "wmv", "flv", "3gp", "ogv",
})
WRITE_BUFFER = 1024 * 1024
STALE_RECEIVING_S = 24 * 3600
STALE_COMPLETE_S = 72 * 3600

router = APIRouter(prefix="/api", tags=["uploads"],
                   dependencies=[Depends(require_auth), Depends(require_csrf)])


def _too_large(settings: Settings) -> ApiError:
    return ApiError(413, f"The file is larger than the {settings.max_upload_mb} MB upload limit.", "too_large")


def parse_content_length(value: str | None, settings: Settings) -> int:
    if value is None:
        raise ApiError(411, "The upload must state its size (Content-Length).", "length_required")
    if not value.strip().isdigit():
        raise ApiError(400, "The upload size header is invalid.", "bad_request")
    length = int(value)
    if length > settings.max_upload_mb * 1024 * 1024:
        raise _too_large(settings)
    if length == 0:
        raise invalid("The file is empty.")
    return length


def clean_filename(header: str | None) -> str:
    """Percent-decode and sanitise ``X-Filename``; enforce the extension allow-list."""
    if not header:
        raise invalid("The upload is missing its file name.")
    name = safe_filename(unquote(header), default="video")
    ext = name.rpartition(".")[2].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise invalid(f"Unsupported file type. Upload a video file ({allowed}).")
    return name


async def write_stream(chunks: AsyncIterator[bytes], dest: Path, *, limit: int, too_large: ApiError) -> int:
    """Write ``chunks`` to ``dest``; abort with ``too_large`` once more than ``limit`` bytes arrive."""
    total = 0
    buf = bytearray()
    fh = await anyio.open_file(dest, "wb")
    try:
        async for chunk in chunks:
            total += len(chunk)
            if total > limit:
                raise too_large
            buf += chunk
            if len(buf) >= WRITE_BUFFER:
                await fh.write(bytes(buf))
                buf.clear()
        if buf:
            await fh.write(bytes(buf))
        await fh.flush()
        await anyio.to_thread.run_sync(os.fsync, fh.wrapped.fileno())
    finally:
        await fh.aclose()
    return total


def _purge_stale_uploads(conn: sqlite3.Connection, storage: Storage) -> None:
    t = db.now()
    rows = conn.execute(
        """SELECT id FROM uploads WHERE project_id IS NULL AND (state='failed'
             OR (state='receiving' AND created_at < ?) OR (state='complete' AND created_at < ?))""",
        (t - STALE_RECEIVING_S, t - STALE_COMPLETE_S)).fetchall()
    for r in rows:
        try:
            storage.delete_tree(storage.upload_dir(r["id"]))
        except (ValueError, OSError):
            continue
        conn.execute("DELETE FROM uploads WHERE id=?", (r["id"],))


def _start_upload(conn: sqlite3.Connection, storage: Storage, settings: Settings, name: str,
                  length: int) -> tuple[str, Path]:
    _purge_stale_uploads(conn, storage)
    storage.require_free(length, settings.min_free_disk_mb, "this upload")
    upload_id = new_id("u")
    udir = storage.upload_dir(upload_id)
    udir.mkdir(parents=True, exist_ok=False)
    final = udir / name
    conn.execute(
        """INSERT INTO uploads(id, filename, path, size_bytes, state, created_at)
           VALUES (?, ?, ?, 0, 'receiving', ?)""",
        (upload_id, name, storage.rel(final), db.now()))
    return upload_id, final


def _abort_upload(conn: sqlite3.Connection, storage: Storage, upload_id: str) -> None:
    try:
        storage.delete_tree(storage.upload_dir(upload_id))
    except (ValueError, OSError):
        pass
    conn.execute("UPDATE uploads SET state='failed' WHERE id=?", (upload_id,))


def _finish_upload(conn: sqlite3.Connection, part: Path, final: Path, upload_id: str, size: int) -> None:
    os.replace(part, final)
    conn.execute("UPDATE uploads SET state='complete', size_bytes=? WHERE id=?", (size, upload_id))


@router.post("/uploads", response_model=UploadOut, status_code=201)
async def upload(request: Request,
                 conn: sqlite3.Connection = Depends(get_conn),
                 settings: Settings = Depends(get_settings),
                 storage: Storage = Depends(get_storage)) -> UploadOut:
    length = parse_content_length(request.headers.get("content-length"), settings)
    name = clean_filename(request.headers.get("x-filename"))
    upload_id, final = await run_in_threadpool(_start_upload, conn, storage, settings, name, length)
    part = final.with_name(final.name + ".part")
    try:
        size = await write_stream(request.stream(), part, limit=min(length, settings.max_upload_mb * 1024 * 1024),
                                  too_large=_too_large(settings))
        if size != length:
            raise ApiError(400, "The upload was incomplete. Please try again.", "upload_incomplete")
        await run_in_threadpool(_finish_upload, conn, part, final, upload_id, size)
    except BaseException:                       # incl. client disconnect and cancellation
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(_abort_upload, conn, storage, upload_id)
        raise
    return UploadOut(id=upload_id, filename=name, size_bytes=size)
